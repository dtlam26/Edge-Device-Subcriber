import ctypes
import sys,os,shutil
import functools
import psutil
import json
import threading,importlib, subprocess
import time
import redis
import cv2
import numpy as np
import platform
import os
from time import sleep
import random



def isCoral():
    if platform.linux_distribution()[0].lower() == 'mendel':
        return True
    return False

def inject_server_info(function):

    def wrap_function(*args, **kwargs):
        with open('./config.json', 'r') as f:
            config = json.load(f)
        kwargs['server'] = config["server"]
        kwargs['port'] = config["port"]
        kwargs['AUTH_KEY'] = config["AUTH"]
        return function(*args, **kwargs)
    return wrap_function

@inject_server_info
def edge_addr_assign(r,name,**kwargs):
    r.client_setname(name)
    current_active = r.client_list()
    current_active.reverse()
    for current_edge in current_active:
        if current_edge['name'] == name:
            edge_addr = current_edge['addr']
            break

    ip = edge_addr.split(":")[0]
    hub_port = edge_addr.split(":")[1]
    if ip.endswith('.0.1'):
        edge_addr = kwargs['server']+":"+hub_port
    return edge_addr

def reload_module(distinct):
    print("Reload...")
    loaded_package_modules = [key for key, value in sys.modules.items() if distinct in str(value)]
    for key in loaded_package_modules:
        print(key)
        del sys.modules[key]

def list_models_available():
    models = os.listdir('./models')
    print("Available models",models)
    return models

def encode_model_with_camera(model_name,camera_index):
    return f"{model_name}_{camera_index}"

def fuse_input_information(camera_list,config,edgetype):
    camera_list[config[0]]["FPS"] = config[1]
    camera_list[config[0]]["threshold"] = config[2]
    camera_list[config[0]]["cropx"] = f"{config[3]};{config[4]}"
    camera_list[config[0]]["cropy"] = f"{config[5]};{config[6]}"
    camera_list[config[0]]["publish"] = config[7]
    if config[8]:
        camera_list[config[0]]["listen"] = [x for x in config[8].split(",")]
    model_folder = encode_model_with_camera(config[9],config[0])
    root_folder = f"./models/{model_folder}"

    if edgetype==1:
        camera_list[config[0]]["model_path"] = os.path.join(root_folder,'model.tflite')
    else:
        if os.path.isfile(os.path.join(root_folder,'trt_graph.bin')):
            camera_list[config[0]]["model_path"] = os.path.join(root_folder,'trt_graph.bin')
            #Inference by trt_cuda
            camera_list[config[0]]["inference_settings"] = 1
        elif os.path.isfile(os.path.join(root_folder,'model.tflite')):
            camera_list[config[0]]["model_path"] = os.path.join(root_folder,'model.tflite')
        else:
            camera_list[config[0]]["model_path"] = os.path.join(root_folder,'model.pb')
            #Inference by tensorflow
            camera_list[config[0]]["inference_settings"] = 2

    if os.path.isdir(os.path.join(root_folder,'execute')):
        camera_list[config[0]]["inference"] = importlib.import_module(".run",package=f"models.{model_folder}.execute")

    camera_list[config[0]]["label_path"] = os.path.join(root_folder,'label.txt')
    camera_list[config[0]]["normalize"] = int(config[10])
    camera_list[config[0]]["image_shape"] = int(config[12])


def draw_on_image(cv2_im,img_orig_shape,bb,label,score,color=(0,200,30),labelon = True):
    x1,y1 = int(bb[0]*img_orig_shape[1]),int(bb[1]*img_orig_shape[0])
    x2,y2 = int(bb[2]*img_orig_shape[1]),int(bb[3]*img_orig_shape[0])
    cv2.rectangle(cv2_im,(x1,y1),(x2,y2),color, thickness=2)
    if labelon:
        cv2.putText(cv2_im, "%s: %.2f" % (label, score), (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,14,255),2)
    return cv2_im

def cvt_xyxy_xywh(bbs):
    tempt = np.empty_like(bbs)
    tempt[:,0] = (bbs[:,0]+bbs[:,2])/2
    tempt[:,1] = (bbs[:,1]+bbs[:,3])/2
    tempt[:,2] = (bbs[:,2]-bbs[:,0])
    tempt[:,3] = (bbs[:,3]-bbs[:,1])
    return tempt

def cvt_xywh_xyxy(bbs):
    tempt = np.empty_like(bbs)
    tempt[:,0] = bbs[:,0]-(bbs[:,2]/2)
    tempt[:,1] = bbs[:,1]-(bbs[:,3]/2)
    tempt[:,2] = bbs[:,0]+(bbs[:,2]/2)
    tempt[:,3] = bbs[:,1]+(bbs[:,3]/2)
    return tempt

class VPN_Connect():

    def __init__(self,benchmark_folder):
        # list of VPN server codes
        self.codeList = ["TR", "US-C", "US", "US-W", "CA", "CA-W",
                    "FR", "DE", "NL", "NO", "RO", "CH", "GB", "HK"]
        self.benchmark_folder = benchmark_folder
        if (os.path.isdir(self.benchmark_folder)):
            shutil.rmtree(self.benchmark_folder)

        os.mkdir(self.benchmark_folder)

    def random_connect(self):
        choiceCode = random.choice(self.codeList)
        # try:
        os.system("windscribe connect " + choiceCode)
        # except:
        #     print("pls check whether you have installed windscribe and login")
        #     # disconnect VPN
        #     self.close()

    def status(self):
        os.system("windscribe status")

    def close(self):
        os.system("windscribe disconnect")

class ThreadidManagement():
    def __init__(self,func=None):
        #grep -r '__NR_gettid' /usr/include/
        # functools.update_wrapper(self, func)
        self.func = func
        self.libc = 'libc.so.6'
        self.cmd = 178
        self.first_mem = 0
        self.last_mem = 0
        ######FASTER
        for key in [178,186,224]:
            tid = ctypes.CDLL(self.libc).syscall(key)
            if key != -1:
                try:
                    psutil.Process(tid)
                    self.cmd = key
                    break
                except:
                    pass

        #####SLOW BUT MORE ACCURATE
        # proc = subprocess.Popen(['grep','-r',"__NR_gettid","/usr/include/"],stdout=subprocess.PIPE)
        # lines = proc.stdout.readlines()
        # print(lines)
        # self.cmd  = int(lines[2].decode("utf-8").split(" ")[-1][:3])


    def gettid(self):
        """Get TID as displayed by htop."""
        tid = ctypes.CDLL(self.libc).syscall(self.cmd)
        return tid

    def immidiate_measure(self,pid,stop_event):
        while(not stop_event.is_set()):
            proc = psutil.Process(pid)
            with proc.oneshot():
                mem_percent = proc.memory_percent("uss")
                if self.first_mem == 0:
                    self.first_mem = mem_percent
                else:
                    self.last_mem = mem_percent


    def __call__(self,*args, **kwargs):
        # def wrapper(*args, **kwargs):
        pid = self.gettid()
        stop_event = threading.Event()
        t = threading.Thread(target=self.immidiate_measure,args=(pid,stop_event))
        t.start()
        results = self.func(*args, **kwargs)
        time.sleep(0.3)
        stop_event.set()
        t.join()
        growth = self.last_mem-self.first_mem
        return results,pid,growth
        # return wrapper
