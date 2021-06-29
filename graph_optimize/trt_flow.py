import cv2
try:
    import tensorrt as trt
    import pycuda.driver as cuda
except:
    print("No Tensort on this device")
import numpy as np

class TrtModel(object):
    """TrtSSD class encapsulates things needed to run TRT SSD."""
    def _preprocess_trt(self,img):
        """Preprocess an image before TRT SSD inferencing."""
        img = cv2.resize(img, (self.input_shape,self.input_shape))
        draw_img = img.copy()
        img = img.astype(np.float32)
        if self.normalize==1:
            img /= 255.0
        elif self.normalize==2:
            img *= (2.0/255.0)
            img -= 1.0
        img = img.transpose((2, 0, 1))
        return img,draw_img

    def _reshape_trt(self,output):
        re_order = output.copy()
        for i in range(self.num_output):
            re_order[i] = output[self.output_order[i]].reshape(self.num_anchors,-1)
        return re_order

    # def _load_plugins(self):
    #     trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self):
        print(f"Loading .... from {self.model_path}")
        with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self):
        host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings = \
            [], [], [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
        return host_inputs, host_outputs, cuda_inputs, cuda_outputs, bindings

    def __init__(self, model_path, anchors, property,output_order):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model_path = model_path
        self.input_shape = property['image_shape']
        self.normalize = property['normalize']
        self.anchors = anchors
        self.num_anchors = self.anchors.shape[0]
        self.output_order = output_order
        # self.cuda_ctx = cuda_ctx
        # cuda.init()
        # device = cuda.Device(0)
        self.cuda_ctx = property['cuda_ctx']
        assert self.cuda_ctx, "Where is your cuda context?"
        self.cuda_ctx.push()
        self.trt_logger = property['trt_logger']

        # self._load_plugins()
        self.engine = self._load_engine()
        assert self.engine, "Seems like the model is conducted in a different CUDA config"
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        try:
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            self.host_inputs, self.host_outputs, self.cuda_inputs, self.cuda_outputs, self.bindings = self._allocate_buffers()
            self.num_input = len(self.host_inputs)
            self.num_output = len(self.host_outputs)
        except Exception as e:
            print(e)
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()


    def __del__(self):
        """Free CUDA memories and context."""
        del self.cuda_outputs
        del self.cuda_inputs
        del self.stream
        # self.cuda_ctx.pop()

    def inference(self, img):
        """Detect objects in the input image."""
        img_resized,draw_img = self._preprocess_trt(img)
        np.copyto(self.host_inputs[0], img_resized.ravel())

        if self.cuda_ctx:
            self.cuda_ctx.push()
        for i in range(self.num_input):
            cuda.memcpy_htod_async(
                self.cuda_inputs[i], self.host_inputs[i], self.stream)
        self.context.execute_async(
            batch_size=1,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        for i in range(self.num_output):
            cuda.memcpy_dtoh_async(
                self.host_outputs[i], self.cuda_outputs[i], self.stream)
        self.stream.synchronize()
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        output = self._reshape_trt(self.host_outputs)
        return output
