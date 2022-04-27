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
import distro
import os
from pathlib import Path
import wget
from collections import defaultdict
import traceback
import multiprocessing
from time import sleep
import pickle
from loguru import logger as LOGGER
import pprint as pp

def BIE(current_bie,training_mean,training_std):
    scaler = 5.5
    z_bie = ((current_bie - training_mean)/training_std).clip(1e-9)
    return np.round((np.log(scaler/(z_bie))-1).clip(0),4)

def swap_xy(x):
    y = np.copy(x)
    for i in range(x.shape[-1]):
        if i%2:
            y[:,i] = x[:,i-1]
        else:
            y[:,i] = x[:,i+1]
    return y

def isCoral():
    if distro.id().lower() == 'mendel':
        return True
    return False

def setup_folder_tree():
    if not os.path.isdir('./videos'):
        os.mkdir('./videos')
    if not os.path.isdir('./models'):
        os.mkdir('./models')

def inject_server_info(function):

    def wrap_function(*args, **kwargs):
        with open('./setting.json', 'r') as f:
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

def no_camera_handle(cuda_ctx):
    LOGGER.info("There is no Camera Available!")
    if cuda_ctx:
        cuda_ctx.pop()

def reload_module(distinct):
    LOGGER.info("Reload...")
    loaded_package_modules = [key for key, value in sys.modules.items() if distinct in str(value)]
    for key in loaded_package_modules:
        LOGGER.info(key)
        del sys.modules[key]

def list_models_available():
    models = os.listdir('./models')
    available = '\n'.join(list(models))
    LOGGER.info(f"Available models: "+available)
    return models

def encode_model_with_camera(model_name,camera_index):
    return f"{model_name}_{camera_index}"

def extract_model2storage(receive,edge_addr,edgetype):
    config = defaultdict(dict)

    index = receive['available'].index(edge_addr)
    update_mode = receive['update_mode'][index]
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
        wget.download(link,model_path)
        # try:
        meta_path = 'http://'+str(Path(receive['file']).with_name('meta.pkl'))
        wget.download(meta_path,os.path.join(root_folder,'meta.pkl'))
        # except:
        #     pass

        if edgetype != 1 and (receive['quantized']<2 or receive['quantized']==6) and suffix !='tflite':
            saved_model_mode = False
            if receive['file'].endswith('_pycuda.pb'):

                LOGGER.warning("Converting through Pub/Sub is not efficient, try to build your own and publish to this device")

                # from graph_optimize.utils import convert_to_local_cuda_driver_config,load_in_out_info
                assert cuda_ctx, "Pls install PyCUDA!"
                # config[int(receive['camera'][index])]["cuda_ctx"] = cuda_ctx
                # config[int(receive['camera'][index])]["trt_logger"] = trt_logger
                wget.download(link.replace(suffix,'json'),model_path.replace(suffix,'json'))

                exe = subprocess.Popen(['chmod', '+x', './graph_optimize/convert_maximize_perfomance.py'],stdout=subprocess.PIPE)
                exe = subprocess.Popen(['python3','./graph_optimize/convert_maximize_perfomance.py',model_path,root_folder,str(receive['image_shape']),
                                        str(receive['quantized'])],stderr=subprocess.PIPE,stdout=subprocess.PIPE)
                LOGGER.info(exe.stderr.readlines())
                LOGGER.info(exe.stdout.readlines())
                # assert len(exe.stderr.readlines()) > 0, [LOGGER.info(e) for e in exe.stderr.readlines()]

            elif receive['file'].endswith('zip'):
                shutil.unpack_archive(model_path, f'{root_folder}/saved_model', 'zip')
                os.remove(model_path)
                model_path = f'{root_folder}/saved_model'
                saved_model_mode = True

        if receive['calib_data']:
            ## Building Calibaration on Edge is not a good idea :(
            from graph_optimize.utils import calib_data

            data_path = os.path.abspath(f"{root_folder}/data.zip")
            wget.download('http://'+receive['calib_data'],data_path)
            shutil.unpack_archive(data_path, f'{root_folder}/data', 'zip')
            os.remove(data_path)
            try:
                """create task in a separate memory as we dont need to share it"""
                p = multiprocessing.Process(target=calib_data,args=(model_path,receive['image_shape'],receive['quantized'],f'{root_folder}/data',saved_model_mode))
                p.start()
                p.join()
                # calib_data(model_path,receive['image_shape'],receive['quantized'],f'{root_folder}/data',saved_model_mode)
            except Exception as e:
                traceback.LOGGER.info_exc()

    if int(receive['quantized']) not in [1,2,3]:
        try:
            anchor_path = os.path.abspath(f"{root_folder}/anchors.npy")
            wget.download(link.replace('.'+suffix,'_anchors.npy'),anchor_path)
        except:
            try:
                wget.download(link.replace('_edgetpu.'+suffix,'_anchors.npy'),anchor_path)
            except Exception as e:
                LOGGER.info(f"This {receive['model_name']} dont need local anchors")

    #LABEL DOWNLOAD
    if receive['label']:
        label_path = os.path.abspath(f"{root_folder}/label.txt")
        if os.path.isfile(label_path):
            os.remove(label_path)
        wget.download('http://'+receive['label'],label_path)


    if "runfile" in receive.keys():
        current_file = receive['runfile'].split('/')[-1]
        runfile_path = os.path.abspath(f"./{root_folder}/{current_file}")
        if os.path.isfile(runfile_path):
            os.remove(runfile_path)
        LOGGER.info('http://'+receive['runfile'])
        wget.download('http://'+receive['runfile'],runfile_path)
        shutil.unpack_archive(runfile_path, f'{root_folder}/execute', 'zip')
    return config,index

def decode_and_store_lastest_config(config,init_config,edgetype):
    try:
        config_meta = json.loads(config)
    except:
        config_meta = config
    with open('config.json', 'w') as f:
        json.dump(config_meta, f)
    return default_config_setting(config_meta,init_config,edgetype)

def default_config_setting(meta,init_config,edgetype):
    if init_config:
        config = init_config
    else:
        config = defaultdict(dict)
    for i in range(len(meta)):
        if "video_select" in list(meta[i].keys()):
            video_path = os.path.abspath(f"./videos/{meta[i]['video_select'].split('/')[-1]}")
            if not os.path.exists(video_path):
                video_url = meta[i]['video_select']
                LOGGER.info("store video:",video_url,video_path,"\n")
                wget.download(video_url,video_path)
            config[i]["capture"] = video_path
            config[i]["video"] = True
            model_folder = meta[i]["running_model_name"]
        else:
            model_folder = encode_model_with_camera(meta[i]["running_model_name"],meta[i]["camera_index"])
            config[i]["capture"] = meta[i]["camera_index"]
            config[i]["video"] = False

        config[i]["FPS"] = meta[i]["FPS"]
        config[i]["threshold"] = meta[i]["threshold"]
        config[i]["cropx"] = f"{meta[i]['x_start']};{meta[i]['x_end']}"
        config[i]["cropy"] = f"{meta[i]['y_start']};{meta[i]['y_end']}"
        config[i]["publish"] = meta[i]["publisher"]
        if meta[i]["listen"]:
            config[i]["listen"] = [x for x in meta[i]["listen"].split(",")]
        else:
            config[i]["listen"] = []

        root_folder = f"./models/{model_folder}"
        if edgetype==1:
            config[i]["model_path"] = os.path.join(root_folder,'model.tflite')
            config[i]["inference_settings"] = 0
        else:
            if os.path.isfile(os.path.join(root_folder,'model.bin')):
                config[i]["model_path"] = os.path.join(root_folder,'model.bin')
                #Inference by trt_cuda
                config[i]["inference_settings"] = 1
            if os.path.isfile(os.path.join(root_folder,'model.engine')):
                config[i]["model_path"] = os.path.join(root_folder,'model.engine')
                #Inference by trt_cuda
                config[i]["inference_settings"] = 1
            elif os.path.isfile(os.path.join(root_folder,'model.tflite')):
                config[i]["model_path"] = os.path.join(root_folder,'model.tflite')
                config[i]["inference_settings"] = 0
            elif os.path.isdir(os.path.join(root_folder,'saved_model')):
                #Inference by tensorflow
                config[i]["model_path"] = os.path.join(root_folder,'saved_model')
                config[i]["inference_settings"] = 3
            else:
                config[i]["model_path"] = os.path.join(root_folder,'model.pb')
                #Inference by tensorflow
                config[i]["inference_settings"] = 2

        if os.path.isdir(os.path.join(root_folder,'execute')):
            config[i]["custom"] = importlib.import_module(".run",package=f"models.{model_folder}.execute")
        else:
            config[i]["custom"] = False

        config[i]["tid"] = 0
        config[i]["load"] = 0
        config[i]["allocate"] = 0

        f = open(f'models/{model_folder}/meta.pkl','rb')
        data = pickle.load(f)

        config[i]["training_mean"] = data["training_mean"]
        config[i]["training_std"] = data["training_std"]
        config[i]["label_path"] = os.path.join(root_folder,'label.txt')
        config[i]["normalize"] = int(meta[i]["normalize"])

        if config[i]["normalize"] == 3:
            config[i]["mean"] = data['mean']
            config[i]["scale"] = data['scale']
        else:
            config[i]["mean"] = []
            config[i]["scale"] = []

        config[i]["image_shape"] = int(meta[i]["input_size"])
        config[i]["quantized"] = int(meta[i]["quantized"])
    LOGGER.info("Current config:")
    pp.pprint(config)
    return config


def draw_on_image(cv2_im,img_orig_shape,bb,label,score,color=(0,200,30),labelon = True):
    x1,y1 = int(bb[0]*img_orig_shape[1]),int(bb[1]*img_orig_shape[0])
    x2,y2 = int(bb[2]*img_orig_shape[1]),int(bb[3]*img_orig_shape[0])
    cv2.rectangle(cv2_im,(x1,y1),(x2,y2),color=color, thickness=2)
    if labelon:
        title = "%s: %.2f" % (label, score)
        labelSize,_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_COMPLEX,2,2)
        cv2.rectangle(cv2_im,(x1,y1),(x1+labelSize[0],y1+labelSize[1]), color=color,thickness=-1)
        cv2.putText(cv2_im,title, (x1,y1+labelSize[1]),cv2.FONT_HERSHEY_SIMPLEX,2,color = (255, 255, 255),thickness=2)
    return cv2_im

def score_image(cv2_im,score_im):
    title = "Image Confidence to Collect: %.2f "%(score_im*100) + '%'
    labelSize,_ = cv2.getTextSize(title,cv2.FONT_HERSHEY_COMPLEX,2,2)
    cv2.rectangle(cv2_im,(0,10),(labelSize[0],10+labelSize[1]), color=(0,0,0),thickness=-1)
    cv2.putText(cv2_im,title, (0,10+labelSize[1]),cv2.FONT_HERSHEY_SIMPLEX,2,color = (255, 255, 255),thickness=2)


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
        #     LOGGER.info("pls check whether you have installed windscribe and login")
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
        # LOGGER.info(lines)
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
