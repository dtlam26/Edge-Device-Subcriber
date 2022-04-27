import wget,importlib,threading,json
import os,sys
import argparse
import base64
import aioredis
import redis
import cv2
import asyncio
import numpy as np
import time
import deployment.run as edge
from deployment.hardware_measurement import SubciberInfo
from deployment.utils import ThreadidManagement, reload_module, edge_addr_assign, draw_on_image
import zipfile


# Define get tid function
threadlooker = ThreadidManagement()




async def listen_others(stream_channels,stop_event):
    aior = await aioredis.create_redis(f"redis://{server}:{port}",password=AUTH_KEY)
    while(True):
        if stop_event:
            break
        for stream in stream_channels:
            print(stream)
            if stream not in share_outputs.keys():
                share_outputs[stream] = []
            try:
                messages = await aior.xrevrange(stream, count=1)
                messages =  list([(stream.encode("utf-8"), *m) for m in messages])
                share_outputs[stream].append(messages)
            except Exception as e:
                print(e)
            if share_outputs[stream] > 10:
                _ = share_outputs.pop(0)
    aior.close()


async def camera_capture(cap,channel,stop_event,interpreter,label,camera_info):
    i = 0
    # buffer = [""] * 8
    total = 0
    inference_total = 0
    send_total = 0

    aior = await aioredis.create_redis(f"redis://{server}:{port}",password=AUTH_KEY)
    camera_info["tid"] = threadlooker.gettid()
    print("connection: ",camera_info["tid"])



    while(cap.isOpened()):
        if stop_event.is_set():
            while(cap.isOpened()):
                cap.release()
            break
        # s = time.time()
        retval, image = cap.read()


        # if retval:
        if retval:
            if i == 0:
                img_orig_shape = image.shape
            i = i + 1

            ss = time.time()
            # for infer in camera_info["custom"]:
            if camera_info["custom"]:
                outputs, obtainable = camera_info["custom"].inference(image,interpreter, label,camera_info["threshold"])
            else:
                outputs, obtainable = edge.inference(image,interpreter, label,camera_info["threshold"])

            inference_total  = inference_total + time.time() - ss

            if len(camera_info["listen"])>0:
                for channel in camera_info["listen"]:
                    if channel in share_outputs.keys():
                        print(share_outputs[channel])


            if obtainable:
                # print(outputs)
                outputs_json = json.dumps(outputs)
                if camera_info["publish"] == True:
                    await aior.xadd(channel+"_pub",{"outputs": outputs_json},max_len=100)
                for (bb,l,s) in zip(outputs[0],outputs[1],outputs[2]):
                    image = draw_on_image(image,img_orig_shape,bb,l,s)


            image = cv2.resize(image,(400,400))
            ret, encoded_img = cv2.imencode('.jpg', image)
            # sss = time.time()
            if obtainable:
                await aior.xadd(channel,{"data": encoded_img.tobytes(),"fps": 1/inference_total, "outputs": outputs_json},max_len=10000) #faster execution
            else:
                await aior.xadd(channel,{"data": encoded_img.tobytes(),"fps": 1/inference_total},max_len=10000)

            # send_total = time.time()-sss + send_total
        # elapse = time.time()-s
        # total = total + elapse
        # i = i +1
        # if i % 50:
        #     print(channel,"complete: ",total/i,"inference speed with communicate: ", inference_total/i,"send_ws: ", send_total/i)
    await aior.flushdb()
    # camera_info["listen"] = []
    # camera_info["declare"] = []
    aior.close()

async def stream_measure(subcriber_hardware_info,edge_addr,stop_event):
    aior = await aioredis.create_redis(f"redis://{server}:{port}",password=AUTH_KEY)
    channel = edge_addr+"_hardwareinfo"
    while(True):
        if stop_event.is_set():
            break
        subcriber_hardware_info.measure()
        results = subcriber_hardware_info.observe
        # print(results)
        await aior.xadd(channel,{"data" : json.dumps(results)},max_len=500) #faster execution
    aior.close()
    while(subcriber_hardware_info.is_alive()):
        subcriber_hardware_info.close()

def stream_redis(edge_addr,cameras,thread_key,thread_list,mainkey):
    # cap_list = {}
    for i in ([x for x in cameras.keys() if cameras[x]["map"] != ""]):

        print("stream from ",i,"with ",cameras[i]["FPS"], " fps")

        cap = cv2.VideoCapture(i)
        cap.set(cv2.CAP_PROP_FPS,cameras[i]["FPS"])
        interpreter, label = edge.load_model(cameras[i])
        channel = edge_addr+"_"+str(i)
        loop = asyncio.new_event_loop()
        stop_event = threading.Event()
        if len(cameras[i]["listen"])>0:
            subcriber_listen = threading.Thread(target=asyncio.run,args=(listen_others(cameras[i]["listen"],stop_event),))
            subcriber_listen.start()
            thread_list.append(subcriber_listen)
        _thread = threading.Thread(target=asyncio.run, args=(camera_capture(cap,channel,stop_event,interpreter,label,cameras[i]),))
        _thread.start()
        print(_thread.getName())
        thread_list.append(_thread)
        thread_key.append(stop_event)
        # asyncio.run(camera_capture(cap_list[i],aior,edge_addr,i))
        # asyncio.create_task(camera_capture(cap, aior,edge_addr,i))

    # SEPARATE HARDWARE STREAM
    loop = asyncio.new_event_loop()
    stop_event = threading.Event()
    subcriber_hardware_info = SubciberInfo(cameras,mainkey)
    _thread = threading.Thread(target=asyncio.run, args=(stream_measure(subcriber_hardware_info,edge_addr,stop_event),))
    _thread.start()
    thread_key.append(stop_event)
    thread_list.append(_thread)

def stop_stream(thread_key,thread_list):
    for thread in thread_key:
        thread.set()
    for thread in thread_list:
        print("If any thread alive?: ",thread.is_alive())
        thread.join()
    print("Final Check")
    for thread in thread_list:
        print("If any thread alive?: ",thread.is_alive())
    for key in camera_list.keys():
        camera_list[key]["tid"] = 0
    return [],[]


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=str, help='edgename')
parser.add_argument('-t', type=str, help='edgetype', default='coral')
parser.add_argument('-p', type=str, help='edgeviewport')
parser.add_argument('-c', type=str, help='cam', default=0)
args = parser.parse_args()
server = '147.46.116.138'
port = 55703
AUTH_KEY = 'UD8fpYG+CJ6mFd2UbRUFkjDa2McgeaffMvsgIG0ZSHUXjRHD2rNXmOzcBDeYlY93nhSiP9mEvWlO9M'

r = redis.StrictRedis(host=server, port=port, db=1, password=AUTH_KEY)
name = f"{args.n}:{args.t}:{args.c}:{args.p}"
edge_addr = edge_addr_assign(r,name,server)

mainkey = os.getpid()
print("LIVE with addr:", edge_addr, " ** not true address", "current pid: ", mainkey)
control_subcriber = r.pubsub()
control_subcriber.ignore_subscribe_messages
control_subcriber.subscribe('deploy')
r.publish("edge_response",json.dumps({"edge": edge_addr,"uptime": int(time.time())}))
# Control subcriber will control when stream rtsp and inference | Interrupt those process for new model deployment

current_file = ""
stop_if_run = False
camera_list = {int(x) : {"map": "","tid":0,"inference":0,"load:":0,"allocate:":0,"listen":[],"publish":False} for x in args.c.split(",")}
thread_key = []
thread_list = []
share_outputs = {}
while(True):
    # try:
    for message in control_subcriber.listen():
        if message and message['data'] != 1:
            receive = message['data'].decode('utf-8')
            receive = json.loads(receive)
            if receive['command'] == 'download':
                if 'tflite' in receive['file'] and edge_addr in receive['available']:

                    index = receive['available'].index(edge_addr)
                    #Stop models running
                    thread_key,thread_list = stop_stream(thread_key,thread_list)
                    #reload_module if neccesary
                    if camera_list[int(receive['camera'][index])]["custom"]:
                        reload_module(camera_list[int(receive['camera'][index])]["model_path"].split("/")[-1].split(".")[0])

                    link = 'http://'+receive['file']

                    # try:
                    # print("Download from :", link)
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 3}))
                    current_file = receive['file'].split('/')[-1]
                    model_as_folder = current_file.split('.')[0]

                    print(model_as_folder)

                    # """remove if file duplicate name, trusted newer model"""
                    #MODEL DOWNLOAD
                    if os.path.exists(f"./model_storage/{current_file}"):
                        os.remove(f"./model_storage/{current_file}")
                    wget.download(link,f"./model_storage/{current_file}")
                    camera_list[int(receive['camera'][index])]["model_path"] = f"./model_storage/{current_file}"
                    camera_list[int(receive['camera'][index])]["map"] = f"Cam {receive['camera'][index]}:{receive['model_name']}\n"

                    #LABEL DOWNLOAD
                    current_file = receive['label'].split('/')[-1]
                    if os.path.exists(f"./label_storage/{current_file}"):
                        os.remove(f"./label_storage/{current_file}")
                    wget.download('http://'+receive['label'],f"./label_storage/{current_file}")
                    camera_list[int(receive['camera'][index])]["label_path"] = f"./label_storage/{current_file}"
                    res = ""
                    for key in camera_list.keys():
                        res = res + camera_list[int(key)]["map"]

                    if "runfile" in receive.keys():
                        current_file = receive['runfile'].split('/')[-1]
                        if os.path.exists(f"./model_execution/{current_file}"):
                            os.remove(f"./model_execution/{current_file}")
                        print('http://'+receive['runfile'])
                        wget.download('http://'+receive['runfile'],f"./model_execution/{current_file}")
                        with zipfile.ZipFile(f"./model_execution/{current_file}", 'r') as z:
                            z.extractall(path='./model_execution/'+model_as_folder)

                        camera_list[int(receive['camera'][index])]["custom"] = importlib.import_module(".run",package=f"model_execution.{model_as_folder}")
                    else:
                        camera_list[int(receive['camera'][index])]["custom"] = 0
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 2,"camera_index": receive['camera'][index], "model_name": res}))
                    # except:
                    #     r.publish("edge_response",json.dumps({"edge": edge_addr,"deployable": 1}))

                    print(camera_list)
            if receive['command'] == 'stream':
                if edge_addr in receive['available']:
                    for config in receive['config']:
                        camera_list[config[0]]["FPS"] = config[1]
                        camera_list[config[0]]["threshold"] = config[2]
                        camera_list[config[0]]["publish"] = config[3]
                        if config[4]:
                            camera_list[config[0]]["listen"] = [x+"_pub" for x in config[4].split(",")]
                    r.publish("edge_response",json.dumps({"edge": edge_addr,"uptime": int(time.time()),"status": 2}))
                    print(camera_list)
                    stream_redis(edge_addr,camera_list,thread_key,thread_list,mainkey)

            if receive['command'] == 'stop':
                thread_key,thread_list = stop_stream(thread_key,thread_list)
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
