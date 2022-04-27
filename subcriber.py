import wget,importlib,json
import os,sys,shutil,glob, subprocess
import argparse
import base64
import redis
import cv2
import numpy as np
import time
import traceback
from loguru import logger as LOGGER
with open('./setting.json', 'r') as f:
    config = json.load(f)

server = config["server"]
port = config["port"]
AUTH_KEY = config["AUTH"]


#UNCOMMENT WHEN PARSING TENSORFLOW FOR JETSON
# try:
#     import tensorflow as tf
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
# except:
#     print("Not limitting GPU Initialize")

from deployment.utils import *
setup_folder_tree()

import deployment.stream as stream_from_edge
import platform

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=str, help='edge name')
parser.add_argument('-t', type=str, help='edge type')
parser.add_argument('-c', type=str, default="", help='camera index')
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

if args.c:
    available_camera = args.c
else:
    videos = glob.glob('/dev/video*')
    assert len(videos)>0, "Can't find the connected Cam"
    available_camera = str([int(c) for c in [i.split('video')[-1] for i in videos] if c.isnumeric()])[1:-1].replace(" ","")

LOGGER.info(f"Cameras: {available_camera}")
r = redis.Redis(host=server, port=port, password=AUTH_KEY)
name = f"{edgename}:{edgetype}:{available_camera}"
edge_addr = edge_addr_assign(r,name)

mainkey = os.getpid()
LOGGER.info("LIVE with addr: " + edge_addr + " ** not true address" + " with current pid: " + str(mainkey))
control_subcriber = r.pubsub()
control_subcriber.ignore_subscribe_messages
control_subcriber.subscribe('deploy')

r.publish("edge_response",json.dumps({"edge": edge_addr,"uptime": int(time.time()), "available_models": list_models_available()}))
# Control subcriber will control when stream rtsp and inference | Interrupt those process for new model deployment

current_file = ""
stop_if_run = False

thread_key = []
thread_list = []
init_config = None
config = None
close = False
while( not close):
    try:
        for message in control_subcriber.listen():
            if message and message['data'] != 1:
                receive = message['data'].decode('utf-8')
                receive = json.loads(receive)
                if receive['command'] == 'download':
                    if edge_addr in receive['available']:
                        LOGGER.info(receive)
                        try:
                            if config:
                                thread_key,thread_list = stream_from_edge.stop_stream(config,thread_key,thread_list)
                            r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 3, "status": 2}))
                            init_config, index = extract_model2storage(receive,edge_addr,edgetype)
                            r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 2, "status": 1, "camera_index": receive['camera'][index], "model_name": receive['model_name']}))

                        except:
                            traceback.print_exc()
                            r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 1}))

                if receive['command'] == 'stream':

                    if edge_addr in receive['available']:
                        # for config in receive['config']:
                        #     fuse_input_information(camera_list,config,edgetype)
                        config = decode_and_store_lastest_config(receive['config'],init_config,edgetype)
                        r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 2}))
                        reload_module("deployment")
                        # stream_from_edge = importlib.import_module(".stream",package=f"deployment")
                        stream_from_edge.stream_redis(r,edge_addr,config,thread_key,thread_list,mainkey)

                if receive['command'] == 'video':
                    if edge_addr in receive['available']:
                        config = decode_and_store_lastest_config(receive['config'],init_config,edgetype)
                        r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 2}))
                        reload_module("deployment")
                        stream_from_edge.stream_redis(r,edge_addr,config,thread_key,thread_list,mainkey)

                if receive['command'] == 'stop':
                    if edge_addr in receive['available']:
                        thread_key,thread_list = stream_from_edge.stop_stream(config,thread_key,thread_list)
                        time.sleep(1)
                        r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 1}))
                        LOGGER.info("STOPPED")
    except:
        traceback.print_exc()
        
        try:
            thread_key,thread_list = stop_stream(thread_key,thread_list)
            r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 0}))
            print("CLOSE")
            close = True
            break
        except:
            print("CLOSE WITH EXCEPTION")
            close = True
            break
