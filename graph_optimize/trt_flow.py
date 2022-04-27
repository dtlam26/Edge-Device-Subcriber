import cv2
import random
import os
import numpy as np
import shutil
import time
import threading
from loguru import logger as LOGGER
from .utils import non_max_suppression, pad2square, unpad_from_square
from deployment.utils import ThreadidManagement, swap_xy, BIE
from collections import namedtuple
from loguru import logger as LOGGER
import gc
#UNCOMMENT WHEN PARSING TENSORFLOW FOR JETSON
# try:
#     import tensorflow as tf
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
#     from tensorflow.python.compiler.tensorrt import trt_convert
# except:
#     trt_convert = None
#     print("No Tensort from Tensorflow")
try:
    import tensorrt as trt
except:
    trt = None
    LOGGER.info("No Tensort on this device")
try:
    import pycuda.driver as cuda
    # import pycuda.autoinit
    # DEVICE = cuda.Device(0)
    # CUDA_CTX = DEVICE.make_context()
except:
    cuda = None
    LOGGER.info("No PyCuda on this device")


class TrtModel():
    def __init__(self,engine_path,config,**kwargs):
        assert trt and cuda, "Can't inference by TRT on this device"
        cuda.init()
        device = cuda.Device(0)
        self.lock = threading.Lock()
        self.cuda_ctx = device.make_context()
        overlapped_attributes = {'engine','inputs','outputs','bindings','stream','context'}.intersection(set(kwargs.keys()))
        self.config = config
        assert not overlapped_attributes, f"Can't overlap attributes: {overlapped_attributes}"
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.channel_first = False
        self.img_shape = (0,0)
        LOGGER.info("Model path at : %s"%(engine_path))
        logger = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(logger)
        #Load model
        trt.init_libnvinfer_plugins(None, "")

        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.engine = trt_runtime.deserialize_cuda_engine(engine_data)
        LOGGER.info("Loaded_model")
        self.inputs,self.outputs,self.bindings,self.stream = self.allocate(self.engine)
        self.context = self.engine.create_execution_context()
        self.cuda_ctx.pop()

    # @ThreadidManagement
    def allocate(self,engine):
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'host_mem', 'device_mem'))
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for index in range(engine.num_bindings):
            name = engine.get_binding_name(index)
            dtype = trt.nptype(engine.get_binding_dtype(index))
            shape = tuple(engine.get_binding_shape(index))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(engine[index]):
                inputs.append(Binding(name, dtype, shape, host_mem, device_mem))
            else:
                outputs.append(Binding(name, dtype, shape, host_mem, device_mem))


        LOGGER.debug(f'Inputs {inputs}')
        if inputs[0].shape[1] == 3:
            self.channel_first = True
            self.img_shape = inputs[0].shape[2:]
        else:
            self.img_shape = inputs[0].shape[1:3]

        LOGGER.debug(f'Outputs {outputs}')

        return inputs,outputs,bindings,stream



    def inference(self,im,batch_size=1):
        self.cuda_ctx.push()
        self.lock.acquire()
        im = self.pre_process(im)
        assert im.shape == self.inputs[0].shape, (im.shape, self.inputs[0].shape)

#         self.inputs['images'] = int(im.data_ptr())
#         self.context.execute_v2(list(self.binding_addrs.values()))
#         y = self.bindings['output'].data

        im = im.astype(self.inputs[0].dtype)

        np.copyto(self.inputs[0].host_mem,im.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device_mem, inp.host_mem, self.stream)

        #execute_async_v2 ignore the batch_size
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host_mem, out.device_mem, self.stream)

        self.stream.synchronize()

        self.cuda_ctx.pop()
        self.lock.release()
        return {out.name: out.host_mem.reshape(out.shape) for out in self.outputs}

    def pre_process(self,im):
        cv2_im_rgb = cv2.resize(pad2square(im),self.img_shape)
        if self.config['normalize'] == 2:
            cv2_im_rgb = (cv2_im_rgb-127.5)/127.5
        elif self.config['normalize'] == 1:
            cv2_im_rgb = cv2_im_rgb/255.0
        elif self.config['normalize'] == 3:
            cv2_im_rgb = (cv2_im_rgb-np.asarray(self.config['mean']))/np.asarray(self.config['scale'])
        if self.channel_first:
            return np.moveaxis(np.expand_dims(cv2_im_rgb,0),-1,1)
        else:
            return np.expand_dims(cv2_im_rgb,0)

    # @LOGGER.catch
    def post_process(self,output,label,**kwargs):
        bbs = []
        scores = []
        diff = None
        for k in list(output.keys()):
            if 'scores' in k:
                scores = output[k]
            elif 'location' in k:
                bbs = output[k]
            elif 'diff' in k:
                diff = output[k]
                mean = self.config['training_mean']
                std = self.config['training_std']
                diff = BIE(diff,mean,std)

        assert len(bbs) and len(scores), "Can't find bbs and scores for mapping, you should write your own postprocess"

        prediction = np.concatenate([bbs,scores],2)[0]
        o,k = non_max_suppression(prediction,conf_thres=self.config['threshold'],object_score=False)

        if len(o):
            o[:,:4] = swap_xy(o[:,:4])
            o = unpad_from_square(o,self.config['width'],self.config['height'])
            output = [o[:,:4].tolist(),[label[int(c)-1] for c in o[:,-1].tolist()],o[:,4].reshape(-1).tolist()]
            if diff is not None:
                output.append(diff)
            return output, True
        else:
            output = [[],[],[]]
            if diff is not None:
                output.append(diff)
            return output, False

    def __del__(self):
        try:
            for inp in self.inputs:
                # inp.host_mem.free()
                inp.device_mem.free()
            for out in self.outputs:
                # out.host_mem.free()
                out.device_mem.free()

            LOGGER.info('Free Allocate GPU Memory')
            while 1:
                try:
                    self.cuda_ctx.detach()
                except:
                    break
        except:
            pass

        del self.context
        del self.engine
        del self.cuda_ctx
        del self.stream
        del self.outputs
        del self.inputs
        gc.collect()


if __name__ == '__main__':
    while True:
        model = TrtModel('/home/dtlam26/Documents/Coral_Project/data/Parking Dataset/models/model.engine',{})
        LOGGER.info("DELETE")
        del model
        time.sleep(5)
