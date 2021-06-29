import wget,importlib,threading,json,asyncio
import os,sys,shutil,subprocess
import argparse
import base64
import redis
import cv2
import numpy as np
import time
import zipfile
import random

with open('./config.json', 'r') as f:
    config = json.load(f)

server = config["server"]
port = config["port"]
AUTH_KEY = config["AUTH"]

from deployment.utils import *
import deployment.stream as stream_from_edge
import platform


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=str, help='edge name')
parser.add_argument('-t', type=str, help='edge type')
parser.add_argument('-c', type=str, help='camera index')
args = parser.parse_args()

def edge_operate(benchmark_folder,name,i,time_list):
    global server
    global port
    global AUTH_KEY
    global stream_from_edge
    os.mkdir(os.path.join(benchmark_folder,str(i)))
    model_storage_folder = os.path.join(benchmark_folder,str(i))
    r = redis.Redis(host=server, port=port, password=AUTH_KEY)
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
    camera_list = {int(x) : {"tid":0,"inference":0,"inference_settings":0,"load:":0,"allocate:":0,"listen":[],"publish":False,"cuda_ctx": cuda_ctx} for x in available_camera.split(",")}
    thread_key = []
    thread_list = []
    test = True
    while(test):
        print("Thread_num: ",i)
        # try:
        for message in control_subcriber.listen():
            if message and message['data'] != 1:
                receive = message['data'].decode('utf-8')
                receive = json.loads(receive)
                if receive['command'] == 'download':
                    if edge_addr in receive['available']:
                        s = time.time()
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
                        root_folder = os.path.join(model_storage_folder,model_as_folder)
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

                                if receive['file'].endswith('_pycuda.pb'):
                                    reload_module("graph_optimize")
                                    from graph_optimize.utils import convert_to_local_cuda_driver_config,load_in_out_info
                                    assert cuda_ctx, "Pls install PyCUDA!"
                                    camera_list[int(receive['camera'][index])]["cuda_ctx"] = cuda_ctx

                                    wget.download(link.replace(suffix,'json'),model_path.replace(suffix,'json'))
                                    info = load_in_out_info(root_folder)
                                    convert_to_local_cuda_driver_config(model_path,info,receive['image_shape'],receive['quantized'],root_folder)

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

                        time_list.append(time.time()-s)
                        r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 2,"camera_index": receive['camera'][index], "model_name": receive['model_name']}))
                        test = False

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


cuda_ctx = None
number_of_fake_edges = 10
benchmark_folder = '../benchmark'
vpn = VPN_Connect(benchmark_folder)

# vpn.random_connect()
time_list = []
thread_list = []
for i in range(number_of_fake_edges):
    if args.n != None:
        edgename = args.n
    else:
        edgename = platform.node()

    edgetype  = 1
    available_camera = "0"
    name = f"{edgename}:{edgetype}:{available_camera}"


    t = threading.Thread(target=edge_operate,args=(benchmark_folder,name,i,time_list))
    t.start()
    thread_list.append(t)

for t in thread_list:
    t.join()
a = 0
for t in time_list:
    a = a +t
print(a/number_of_fake_edges)

# vpn.close()
