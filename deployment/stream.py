import sys
import subprocess
import psutil
import json
import threading
import time
import cv2
import numpy as np
import asyncio
import aioredis
from deployment.utils import *
import deployment.run as edge
from deployment.hardware_measurement import SubciberInfo, stream_measure
from deployment.logic_match import MatchingLogic, regress_view
import traceback
from loguru import logger as LOGGER
from matplotlib import cm
COLORS = (cm.get_cmap('Dark2', 8).colors[:,:3]*255).astype(np.int32).tolist()

share_outputs = {}
threadlooker = ThreadidManagement()
logic_match = MatchingLogic()
parent_controller = None

@LOGGER.catch
@inject_server_info
async def camera_capture(channel,interpreter,label,stop_event,camera_info,**kwargs):
    i = 0
    # buffer = [""] * 8
    total = 0
    inference_total = 0
    send_total = 0
    global threadlooker
    global share_outputs
    global logic_match
    global parent_controller
    # interpreter, label = edge.load_model(camera_info)
    aior = await aioredis.create_redis(f"redis://{kwargs['server']}:{kwargs['port']}",password=kwargs['AUTH_KEY'])
    camera_info["tid"] = threadlooker.gettid()
    LOGGER.info("connection: %s at %s"%(channel,camera_info["tid"]))
    listener = True if (len(camera_info["listen"])>0) else False
    old_out = []
    cropx = camera_info["cropx"].split(";")
    cropy = camera_info["cropy"].split(";")
    cropx_start,cropx_end = float(cropx[0]),float(cropx[1])
    cropy_start,cropy_end = float(cropy[0]),float(cropy[1])

    # LOGGER.info("stream from camera",i,"with ",cameras_config[i]["FPS"], " fps")
    if camera_info["inference_settings"] == 0:
        cap = cv2.VideoCapture(camera_info["capture"])
    else:
        if camera_info["video"]:
            cap = cv2.VideoCapture(camera_info["capture"])
        else:
            cap = cv2.VideoCapture(camera_info["capture"], cv2.CAP_V4L)

    camera_info['width'] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    camera_info['height'] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.set(cv2.CAP_PROP_FPS,camera_info["FPS"])

    color_dict = {l:COLORS[c] for c,l in label.items()}
    try:
        frame_counter = 0
        while(cap.isOpened() or not stop_event.is_set()):
            frame_counter += 1
            #Loop for video
            if camera_info["video"]:
                if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT)-1:
                    frame_counter = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    frame_counter += 1

            if stop_event.is_set():
                while(cap.isOpened()):
                    cap.release()
                break
            # s = time.time()
            retval, image = cap.read()
            response_text = ""

            # if retval:
            if retval:
                if i == 0:
                    img_orig_shape = image.shape
                    cropx_start,cropx_end = int(cropx_start*img_orig_shape[1]),int(cropx_end*img_orig_shape[1])
                    cropy_start,cropy_end = int(cropy_start*img_orig_shape[0]),int(cropy_end*img_orig_shape[0])
                    image = image[cropy_start:cropy_end,cropx_start:cropx_end]
                    img_orig_shape = image.shape
                else:
                    image = image[cropy_start:cropy_end,cropx_start:cropx_end]
                i = i + 1

                ss = time.time()
                # for infer in camera_info["custom"]:
                if camera_info["custom"] and camera_info["inference_settings"] == 0:
                    outputs, obtainable = camera_info["custom"].inference(image,interpreter, label,camera_info)
                    LOGGER.info(outputs)
                else:
                    if camera_info["inference_settings"] == 0:
                        outputs, obtainable = edge.inference(image,interpreter, label,camera_info)
                    elif camera_info["inference_settings"] == 1:
                        outputs = interpreter.inference(image)
                        if camera_info["custom"]:
                            outputs, obtainable = camera_info["custom"].post_process(outputs,label)
                        else:
                            outputs, obtainable = interpreter.post_process(outputs,label)
                    elif camera_info["inference_settings"] == 3:
                        outputs, obtainable = interpreter.inference(image)

                inference_total  = time.time() - ss

                if obtainable:
                    # LOGGER.info(outputs)
                    outputs_json = json.dumps(outputs)
                    if listener:
                        # outputs[0] = local_match.filter(np.asarray(outputs[0]))
                        logic_match.store_state(outputs[0],outputs[1],outputs[2])
                else:
                    if listener:
                        logic_match.store_state([],[],[])

                match = False
                t = None
                if listener:
                    matches_list = {}
                    publisher_output_concatenate = [[],[],[],[]]
                    stream_workers = []
                    relation_bb_to_img = {'index':[],'background':[],'camera':[]}
                    for stream in camera_info["listen"]:
                        if stream in share_outputs.keys():
                            if len(share_outputs[stream]) > 0:
                                publisher_output = share_outputs[stream][-1]
                                if abs(time.time()-share_outputs[stream][-1][1]/1000) < 2.5/camera_info["FPS"]:
                                    if camera_info["share_view"]:
                                        publisher_output = share_outputs[stream][-1]
                                        if logic_match.is_share_view_read(stream):
                                            match = True
                                            t = threading.Thread(target=logic_match.predict,args=(publisher_output[0],stream,publisher_output_concatenate,relation_bb_to_img,publisher_output[-1]))
                                            t.start()
                                            stream_workers.append(t)
                                            # publisher_output[0][0] = logic_match.predict(np.asarray(publisher_output[0][0]),stream)
                                            # output_pub = publisher_output[0]
                                            # output_pub[0] = logic_match.filter(output_pub[0])
                                        else:
                                            info = stream+"_match"
                                            if obtainable:
                                                await aior.xadd(info,{"outputs": outputs_json,"inputs":json.dumps(publisher_output[0])})
                                            if await (aior.xlen(info))>100:
                                                LOGGER.info("Mapping")
                                                await regress_view(info,stream)
                                                logic_match.store_matching_model(stream)
                                    else:
                                        if camera_info["custom"]:
                                            response_text = camera_info["custom"].process_input(publisher_output[0],outputs,obtainable)
                                        else:
                                            response_text = edge.process_input(publisher_output[0],outputs,obtainable)

                                    # for (bb,l,s) in zip(output_pub[0],output_pub[1],output_pub[2]):
                                    #     image = draw_on_image(image,img_orig_shape,bb,l,s,color=(250,100,12))
                                #     LOGGER.info("match successful")
                                # else:
                                    # LOGGER.info("cant match, it is late")
                    # for t in stream_workers:
                    #     t.join()
                    if obtainable and camera_info["share_view"] and match:
                        output_pub,mapped_outputs = logic_match.regress_multiple_label_by_ios(publisher_output_concatenate,outputs)
                    # else:
                    #     if len(publisher_output_concatenate[0]>0):
                    #         logic_match.store_state(publisher_output_concatenate[0],publisher_output_concatenate[1],publisher_output_concatenate[2])
                        logic_match.compare_previous_state(threshold=0.4)

                        t = threading.Thread(target=edge.send_back_results,args=(parent_controller,channel,image.copy(),logic_match.matches.copy(),relation_bb_to_img.copy(),publisher_output_concatenate.copy(),logic_match.output_to_json()))
                        t.start()

                        if len(logic_match.bbs) > 0:
                            for (bb,l,s) in zip(logic_match.bbs,logic_match.labels,logic_match.scores):
                                image = draw_on_image(image,img_orig_shape,bb,l,s)
                        image = logic_match.draw_tracks(image,img_orig_shape)
                        for key in logic_match.remove_key:
                            state, bool = logic_match.in_out(key)
                            if bool:
                                logic_match.labels.append(state)
                        logic_match.remove()


                # if listener and camera_info["share_view"] and match:
                    # for key in logic_match.previous_output_sub.keys():
                    #     obj = logic_match.previous_output_sub[key]
                    #     for bb in obj:
                    #         image = draw_on_image(image,img_orig_shape,bb,None,None,color=(0,0,255),labelon=False)
                    # logic_match.compare_previous_state(threshold=0.4)
                    # if len(logic_match.bbs) > 0:
                    #     for (bb,l,s) in zip(logic_match.bbs,logic_match.labels,logic_match.scores):
                    #         image = draw_on_image(image,img_orig_shape,bb,l,s)
                    # image = logic_match.draw_tracks(image,img_orig_shape)
                    # for key in logic_match.remove_key:
                    #     state, bool = logic_match.in_out(key)
                    #     if bool:
                    #         logic_match.labels.append(state)
                    # logic_match.remove()
                    # outputs_json = json.dumps(logic_match.output_to_json())

                    if not camera_info["share_view"]:
                        y0, dy = 50, 20
                        for i, line in enumerate(response_text.split('\n')):
                            y = y0 + i*dy
                            cv2.rectangle(image,(30,y),(500,y-dy),(255,255,255),-1)
                            cv2.putText(image, line, (50, y ), cv2.FONT_HERSHEY_PLAIN, 2, 2)
                    else:
                        if obtainable:
                            for (bb,l,s) in zip(outputs[0],outputs[1],outputs[2]):
                                image = draw_on_image(image,img_orig_shape,bb,l,s)
                else:
                    if obtainable:
                        # for old in old_out:
                        #     for (bb,l,s) in zip(old[0],old[1],old[2]):
                        #         image = draw_on_image(image,img_orig_shape,bb,l,s)

                        for (bb,l,s) in zip(outputs[0],outputs[1],outputs[2]):
                            image = draw_on_image(image,img_orig_shape,bb,l,s,color=color_dict[l])
                        # old_out.append(outputs)
                if len(outputs)>3: #Contain Score for IMAGE
                    score_image(image,outputs[-1])

                image = cv2.resize(image,(int(camera_info['width']/2),int(camera_info['height']/2)))
                ret, encoded_img = cv2.imencode('.jpg', image)
                # sss = time.time()
                if obtainable :
                    if camera_info["publish"]:
                        await aior.xadd(channel,{"data": encoded_img.tobytes(),"fps": 1/inference_total, "outputs": outputs_json, "flag": 1},max_len=10000)

                if not obtainable or not camera_info["publish"]:
                    await aior.xadd(channel,{"data": encoded_img.tobytes(),"fps": 1/inference_total, "flag": 1 if obtainable else 0},max_len=10000)\

                LOGGER.info("Published")
                # send_total = time.time()-sss + send_total
            # elapse = time.time()-s
            # total = total + elapse
            # i = i +1
            # if i % 50:
            #     LOGGER.info(channel,"complete: ",total/i,"inference speed with communicate: ", inference_total/i,"send_ws: ", send_total/i)
        await aior.flushdb()
        cap.release()
        aior.close()
        del interpreter
    except:
        traceback.print_exc()
        stop_event.set()

def stream_redis(r,edge_addr,cameras_config,thread_key,thread_list,mainkey):
    # cap_list = {}
    # for i in ([x for x in cameras_config.keys() if cameras_config[x]["map"] != ""]):
    global parent_controller
    parent_controller = r
    global share_outputs
    model_dict = {}
    for i in (cameras_config.keys()):
        if cameras_config[i]['model_path'] in model_dict:
            interpreter, label = model_dict[cameras_config[i]['model_path']]
        else:
            interpreter, label = edge.load_model(cameras_config[i])
            model_dict[cameras_config[i]['model_path']] = (interpreter,label)

        channel = edge_addr+"_"+str(i)
        loop = asyncio.new_event_loop()
        stop_event = threading.Event()
        if len(cameras_config[i]["listen"])>0:
            subcriber_listen = threading.Thread(target=loop.run_until_complete,args=(edge.listen_others(cameras_config[i]["listen"],stop_event,share_outputs),))
            subcriber_listen.start()
            thread_list.append(subcriber_listen)
        _thread = threading.Thread(target=loop.run_until_complete, args=(camera_capture(channel,interpreter,label,stop_event,cameras_config[i]),))
        _thread.daemon = True
        _thread.start()
        LOGGER.info(_thread.getName())
        thread_list.append(_thread)
        thread_key.append(stop_event)
        # asyncio.run(camera_capture(cap_list[i],aior,edge_addr,i))
        # asyncio.create_task(camera_capture(cap, aior,edge_addr,i))
    r.publish("edge_response",json.dumps({"edge": edge_addr,"status": 3}))
    # SEPARATE HARDWARE STREAM
    # if isCoral():
    loop = asyncio.new_event_loop()
    stop_event = threading.Event()
    subcriber_hardware_info = SubciberInfo(cameras_config,mainkey)
    _thread = threading.Thread(target=loop.run_until_complete, args=(stream_measure(subcriber_hardware_info,edge_addr,stop_event),))
    _thread.start()
    thread_key.append(stop_event)
    thread_list.append(_thread)
    # LOGGER.info(cameras_config)

def stop_stream(camera_list,thread_key,thread_list):
    for thread in thread_key:
        thread.set()
    for thread in thread_list:
        LOGGER.info(f"If any thread alive?: {thread.is_alive()}")
        thread.join()
    LOGGER.info("Final Check")
    for thread in thread_list:
        LOGGER.info(f"If any thread alive?: {thread.is_alive()}")
    for key in camera_list.keys():
        camera_list[key]["tid"] = 0
        # camera_list[key]["cuda_ctx"].pop()
    return [],[]
