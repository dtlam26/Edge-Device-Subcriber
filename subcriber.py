import wget,importlib,threading,json,asyncio
import os,sys,shutil,glob, subprocess
import argparse
import base64
import redis
import cv2
import numpy as np
import time
import zipfile


with open('./config.json', 'r') as f:
    config = json.load(f)

server = config["server"]
port = config["port"]
AUTH_KEY = config["AUTH"]

from deployment.utils import *
import deployment.stream as stream_from_edge
import platform
try:
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    cuda_ctx = device.make_context()
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
except:
    cuda_ctx = None
    TRT_LOGGER = None

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=str, help='edge name')
parser.add_argument('-t', type=str, help='edge type')
parser.add_argument('-c', type=str, help='camera index')
args = parser.parse_args()

if args.n != None:
    edgename = args.n
else:
    edgename = platform.node()

if args.t != None:
    edgetype = args.t
else:
    if isCoral():
        edgetype = 1
    else:
        edgetype = 2

if args.c != None:
    available_camera = args.c
else:
    videos = glob.glob('/dev/video*')
    assert len(videos)>0, "There is no Camera Available!"
    available_camera = str([int(c) for c in [i.split('video')[-1] for i in videos] if c.isnumeric()])[1:-1].replace(" ","")

r = redis.Redis(host=server, port=port, password=AUTH_KEY)
name = f"{edgename}:{edgetype}:{available_camera}"
edge_addr = edge_addr_assign(r,name)

mainkey = os.getpid()
print("LIVE with addr:", edge_addr, " ** not true address", "current pid: ", mainkey)
control_subcriber = r.pubsub()
control_subcriber.ignore_subscribe_messages
control_subcriber.subscribe('deploy')

r.publish("edge_response",json.dumps({"edge": edge_addr,"uptime": int(time.time()), "available_models": list_models_available()}))
# Control subcriber will control when stream rtsp and inference | Interrupt those process for new model deployment

current_file = ""
stop_if_run = False
camera_list = {int(x) : {"tid":0,"inference":0,"inference_settings":0,"load:":0,"allocate:":0,"listen":[],"publish":False,"cuda_ctx": cuda_ctx, "trt_logger": TRT_LOGGER} for x in available_camera.split(",")}
# camera_list = { 0: {"map": "","tid":0,"inference":0,"load:":0,"allocate:":0,"listen":[edge_addr+"_1"],"publish":True,"share_view":True,\
#                     "FPS": 10, "threshold": 0.5, "cropx": "0.35;0.9", "cropy": "0;0.95", "normalize": 2,\
#                     "label_path":"./label_storage/top_head.txt","model_path":"./model_storage/head_best_edgetpu.tflite"},\
#                 1: {"map": "","tid":0,"inference":importlib.import_module(".run",package=f"model_execution.face_mask_detector_tpu"),"load:":0,"allocate:":0,\
#                     "FPS": 10, "threshold": 0.6, "cropx": "0;1", "cropy": "0;1", "normalize": 1,\
#                     "listen":[],"publish":True,"share_view":False,"label_path":"./label_storage/masks.txt","model_path":"./model_storage/face_mask_detector_tpu.tflite"}}
                # 2: {"map": "","tid":0,"inference":0,"load:":0,"allocate:":0,"listen":[],"publish":False,\
                #     "FPS": 10, "threshold": 0.1, "cropx": "0;1", "cropy": "0;1",\
                #     "label_path":"./label_storage/labels.txt","model_path":"./model_storage/head_best_edgetpu.tflite"}}
thread_key = []
thread_list = []

while(True):
    # try:
    for message in control_subcriber.listen():
        if message and message['data'] != 1:
            receive = message['data'].decode('utf-8')
            receive = json.loads(receive)
            if receive['command'] == 'download':
                if edge_addr in receive['available']:
                    print(receive)
                    convert_thread = None
                    index = receive['available'].index(edge_addr)
                    update_mode = receive['update_mode'][index]
                    # print(update_mode)
                    #Stop models running
                    thread_key,thread_list = stream_from_edge.stop_stream(camera_list,thread_key,thread_list)
                    #reload_module if neccesary
                    if camera_list[int(receive['camera'][index])]["inference"] != 0:
                        reload_module(camera_list[int(receive['camera'][index])]["model_path"].split("/")[-1].split(".")[0])
                    # try:
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 3}))
                    suffix = (receive['file'].split('/')[-1]).split('.')[-1]
                    file_name = (receive['file'].split('/')[-1]).split('.')[0]
                    model_as_folder = encode_model_with_camera(receive['model_name'],receive['camera'][index])
                    root_folder = os.path.join('./models',model_as_folder)
                    link = 'http://'+receive['file']
                    if not update_mode:
                        if os.path.isdir(root_folder):
                            shutil.rmtree(root_folder)
                        os.mkdir(root_folder)
                        #MODEL DOWNLOAD
                        model_path = os.path.abspath(f"{root_folder}/model.{suffix}")
                        print(root_folder,link)
                        wget.download(link,model_path)
                        if edgetype != 1 and (receive['quantized']<2 or receive['quantized']==5) and suffix !='tflite':
                            # if receive['file'].endswith('.uff'):
                            #     from graph_optimize.utils import convert_to_local_cuda_driver_config,load_in_out_info
                            #     wget.download(link.replace('uff','json'),model_path.replace('uff','json'))
                            #     info = load_in_out_info(model_path.replace('uff','json'))
                            if receive['file'].endswith('_pycuda.pb'):
                                reload_module("graph_optimize")
                                # from graph_optimize.utils import convert_to_local_cuda_driver_config,load_in_out_info
                                assert cuda_ctx, "Pls install PyCUDA!"
                                camera_list[int(receive['camera'][index])]["cuda_ctx"] = cuda_ctx
                                # import pycuda.driver as cuda
                                # cuda.init()
                                # device = cuda.Device(0)
                                # cuda_ctx = device.make_context()
                                # cuda_ctx.push()
                                wget.download(link.replace(suffix,'json'),model_path.replace(suffix,'json'))
                                # info = load_in_out_info(root_folder)
                                # convert_to_local_cuda_driver_config(model_path,info,receive['image_shape'],receive['quantized'],root_folder,camera_list[int(receive['camera'][index])]['trt_logger'])
                                exe = subprocess.Popen(['chmod', '+x', './graph_optimize/convert_maximize_perfomance.py'],stdout=subprocess.PIPE)
                                exe = subprocess.Popen(['python3','./graph_optimize/convert_maximize_perfomance.py',model_path,root_folder,str(receive['image_shape']),
                                                        str(receive['quantized'])],stderr=subprocess.PIPE,stdout=subprocess.PIPE)
                                print(exe.stderr.readlines())
                                print(exe.stdout.readlines())
                                # assert len(exe.stderr.readlines()) > 0, [print(e) for e in exe.stderr.readlines()]


                    if int(receive['quantized']) not in [2,3]:
                        try:
                            anchor_path = os.path.abspath(f"{root_folder}/anchors.npy")
                            wget.download(link.replace('.'+suffix,'_anchors.npy'),anchor_path)
                        except Exception as e:
                            print(e)
                            print("This is not a model from the framework")

                    #LABEL DOWNLOAD
                    label_path = os.path.abspath(f"{root_folder}/label.txt")
                    if os.path.isfile(label_path):
                        os.remove(label_path)
                    wget.download('http://'+receive['label'],label_path)

                    if "runfile" in receive.keys():
                        current_file = receive['runfile'].split('/')[-1]
                        runfile_path = os.path.abspath(f"./{root_folder}/{current_file}")
                        if os.path.isfile(runfile_path):
                            os.remove(runfile_path)
                        print('http://'+receive['runfile'])
                        wget.download('http://'+receive['runfile'],runfile_path)
                        with zipfile.ZipFile(runfile_path, 'r') as z:
                            z.extractall(path=f'{root_folder}/execute')


                    # if convert_thread is not None:
                    #     convert_thread.join()
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 2,"camera_index": receive['camera'][index], "model_name": receive['model_name']}))

                    # except:
                    #     r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 1}))

            if receive['command'] == 'stream':
                """
                config orders:

                camera_index,FPS,threshold,x_start,x_end,y_start,y_end,publisher,listen,
                model_name,normalize,quantize_level,image_shape,active_ip
                """

                if edge_addr in receive['available']:
                    for config in receive['config']:
                        fuse_input_information(camera_list,config,edgetype)
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 2}))
                    reload_module("deployment")
                    stream_from_edge = importlib.import_module(".stream",package=f"deployment")
                    print(camera_list)
                    stream_from_edge.stream_redis(r,edge_addr,camera_list,thread_key,thread_list,mainkey)

            if receive['command'] == 'stop':
                if edge_addr in receive['available']:
                    thread_key,thread_list = stream_from_edge.stop_stream(camera_list,thread_key,thread_list)
                    time.sleep(1)
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 1}))
                    print("STOPPED")
    # except:
    #     try:
    #         thread_key,thread_list = stop_stream(thread_key,thread_list)
    #         r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 0}))
    #         print("CLOSE")
    #         break
    #     except:
    #         print("CLOSE WITH EXCEPTION")
    #         break
