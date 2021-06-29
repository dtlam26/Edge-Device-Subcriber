import psutil
import json
import threading
import aioredis
from deployment.utils import inject_server_info

@inject_server_info
async def stream_measure(subcriber_hardware_info,edge_addr,stop_event,**kwargs):
    aior = await aioredis.create_redis(f"redis://{kwargs['server']}:{kwargs['port']}",password=kwargs['AUTH_KEY'])
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

class SubciberInfo():
    def __init__(self,cameras,mainkey):
        self.observe = {}
        self.active_threads = []
        self.mainkey = mainkey
        current_ram = psutil.virtual_memory()
        freq = psutil.cpu_freq()
        self.cameras = cameras
        self.num_cpus = psutil.cpu_count()
        self.observe["config"] = {"cpus": self.num_cpus,"memory": round(current_ram.total/10e8,1),\
                                    "freq_max":round(freq.max/10e2,1), "freq_min":round(freq.min/10e2,1)}
        self.observe["tasks_info"] = []
        self.inference_thread = {-1:psutil.Process(self.mainkey)}
        while(True):
            for i in cameras.keys():
                if cameras[i]["tid"] != 0:
                    self.inference_thread[i] = psutil.Process(cameras[i]["tid"])
            if len(self.inference_thread.keys())-1==len(cameras.keys()):
                # print("Prepare for measurement: ",cameras)
                break

    def measure(self):
        self.observe["tasks_info"] = []
        self.active_threads = []
        for proc_key in self.inference_thread.keys():
            t= threading.Thread(target=self.single_process_measure,args=([proc_key]))
            self.active_threads.append(t)
            t.start()
        for t in self.active_threads:
            t.join()

    def close(self):
        for t in self.active_threads:
            t.join()
        self.inference_thread = {}

    def is_alive(self):
        for t in self.active_threads:
            if t.is_alive():
                return True
        return False

    def single_process_measure(self,key):
        c = self.inference_thread[key]
        cam = ""
        if key == -1:
            cam = "Main"
        else:
            cam = f"Cam {key}"
        try:
            with c.oneshot():
                mem = c.memory_info()
                mem_percent = c.memory_percent("uss")
                cpu_time = c.cpu_times()  # return cached value
                cpu_percent = c.cpu_percent()/self.num_cpus  # return cached value
                num = c.cpu_num()
            analysis = {"cam":cam,"num":num,"cpu_percent": round(cpu_percent,1),"user":round(cpu_time.user,1),\
                            "system":round(cpu_time.system,1),"iowait":round(cpu_time.iowait,1),\
                            "rss": round(mem.rss/10e9,3),"vms": round(mem.vms/10e9,3), "ram_percent": round(mem_percent,1)}
            if key != -1:
                analysis["load"] = self.cameras[key]["load"]
                analysis["allocate"] = self.cameras[key]["allocate"]
            self.observe["tasks_info"].append(analysis)
        except:
            print("Exit Process")
