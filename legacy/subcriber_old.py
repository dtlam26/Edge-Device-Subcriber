import redis
import sys
import wget
import json
import run
import stream
import os
import threading
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', type=str, help='edgename')
parser.add_argument('-t', type=str, help='edgetype', default='coral')
parser.add_argument('-p', type=str, help='edgeviewport')
parser.add_argument('-c', type=int, help='cam', default=0)
args = parser.parse_args()


server = '147.46.116.138'
AUTH_KEY = 'UD8fpYG+CJ6mFd2UbRUFkjDa2McgeaffMvsgIG0ZSHUXjRHD2rNXmOzcBDeYlY93nhSiP9mEvWlO9M'
r = redis.StrictRedis(host=server, port=55703, db=2, password=AUTH_KEY)
name = f"{args.n}:{args.t}:{args.c}:{args.p}"
r.client_setname(name)
current_active = r.client_list()
current_active.reverse()
for edge in current_active:
    if edge['name'] == r.client_getname():
        edge_addr = edge['addr']
        break
ip = edge_addr.split(":")[0]
port = edge_addr.split(":")[1]
if ip.endswith('.0.1'):
    edge_addr = server+":"+port
print("LIVE with addr:", edge_addr)
p = r.pubsub()
p.ignore_subscribe_messages
p.subscribe('deploy')

current_file = ""
stop_if_run = False
while(True):
    for message in p.listen():
        if message and message['data'] != 1:
            receive = message['data'].decode('utf-8')
            receive = json.loads(receive)
            if receive['command'] == 'download':
                if 'tflite' in receive['file'] and edge_addr in receive['available']:
                    if stop_if_run:
                        while(not m.stopped):
                            m.stop()
                        stop_if_run = False

                    link = 'http://'+receive['file']
                    print("Download from :", link)
                    current_file = receive['file'].split('/')[-1]
                    # """remove if file duplicate name, trusted newer model"""
                    if os.path.exists(f"model_storage/{current_file}"):
                        os.remove(f"model_storage/{current_file}")
                    wget.download(link,f"model_storage/{current_file}")
                    if os.path.exists("label.txt"):
                        os.remove("label.txt")
                    wget.download('http://'+receive['label'])
            if receive['command'] == 'run' and edge_addr in receive['available']:
                m = run.StoppableDetection(f'model_storage/{current_file}',0.1,camera=args.c)
                m.start()
                stop_if_run = True

            if receive['command'] == 'stream' and edge_addr in receive['available']:
                m = stream.StoppableDetection(f'model_storage/{current_file}',0.1,camera=args.c)
                m.start()
                stop_if_run = True
