import aioredis
import numpy as np
import sklearn
import asyncio
import json
import pickle
server = "147.46.116.138"
port = 55703
AUTH_KEY = "UD8fpYG+CJ6mFd2UbRUFkjDa2McgeaffMvsgIG0ZSHUXjRHD2rNXmOzcBDeYlY93nhSiP9mEvWlO9M"
async def read_stream():
    aior = await aioredis.create_redis(f"redis://{server}:{port}", password=AUTH_KEY)
    message = await aior.xrange(server+"_match")
    return message
cam1 = []
cam0 = []
messages = asyncio.run(read_stream())
for m in messages:
    num_key = len(m[1].keys())
    if num_key ==2:
        v_0 = []
        v_1 = []
        for k, v in m[1].items():
            v = json.loads(v.decode("utf-8"))[0]
            if k.decode("utf-8") == "cam0":
                v_0 = v
            if k.decode("utf-8") == "cam1":
                v_1 = v
        if len(v_0) > 0 and len(v_1)>0 and len(v_0) == len(v_1):
            cam0.append(v_0)
            cam1.append(v_1)

print(len(cam0),len(cam1))
cam0 = np.concatenate(cam0,axis=0)
cam1 = np.concatenate(cam1,axis=0)
print(cam0.shape,cam1.shape)
# tempt = np.empty_like(cam0)
# cam = [cam0,cam1]
# for i,c in enumerate(cam):
#     tempt[:,0] = (c[:,0]+c[:,2])/2
#     tempt[:,1] = (c[:,1]+c[:,3])/2
#     tempt[:,2] = (c[:,2]-c[:,0])
#     tempt[:,3] = (c[:,3]-c[:,1])
#     cam[i] = tempt
from sklearn.linear_model import LinearRegression
reg = LinearRegression(n_jobs=4).fit(cam1, cam0)
print(reg.predict(cam1[250:260]),cam0[250:260])
filename = 'model.sav'
pickle.dump(reg, open(filename, 'wb'))
