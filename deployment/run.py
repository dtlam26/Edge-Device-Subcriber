import numpy as np
import os
from deployment.utils import ThreadidManagement, inject_server_info, isCoral
if isCoral():
    import tflite_runtime.interpreter as tflite
    from tflite_runtime.interpreter import load_delegate
else:
    try:
        import tensorflow.lite as tflite
        import graph_optimize.tensorflow_flow as tf_flow
    except:
        print("There are no Tensorflow on this Device")
    try:
        from graph_optimize.utils import *
        from graph_optimize.trt_flow import TrtModel
    except:
        print("There are no TensorRT on this Device")
import aioredis,json
import cv2
import concurrent.futures
import json
import base64


# @ThreadidManagement
def inference(cv2_im,interpreter,label,property):
    # print(camera_info["tid"])
    # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    cv2_im_rgb = cv2.resize(cv2_im, (input_shape[1],input_shape[2]))
    cv2_im_rgb = normalize_input(cv2_im_rgb,input_details,property['normalize'])
    interpreter.set_tensor(input_details[0]['index'], cv2_im_rgb)
    # Test the model on random input data.

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data1 = interpreter.get_tensor(output_details[0]['index'])[0]
    output_data2 = interpreter.get_tensor(output_details[1]['index'])[0]
    output_data3 = interpreter.get_tensor(output_details[2]['index'])[0]
    output_data4 = interpreter.get_tensor(output_details[3]['index'])[0]
    # print(output_data1,output_data2,output_data3,output_data4)
    bounding_boxes = []
    labels = []
    scores = []
    obtainable = False
    for bb,score,cls in zip(output_data1,output_data3,output_data2):
        if score > property['threshold']:
            # bb = bb.clip(min=0,max=1).tolist()
            bb = bb.tolist()
            score = score.tolist()
            obtainable = True
            bounding_boxes.append([bb[1],bb[0],bb[3],bb[2]])
            labels.append(label[int(cls)])
            scores.append(score)
    return list([bounding_boxes,labels,scores]), obtainable


def process_input(publisher_outputs,current_ouputs,obtainable):
    response_text = f"{len(current_ouputs[0])} available seats \n At entrance: \n"
    for txt in publisher_outputs[1]:
        response_text = response_text + f"{txt}\n"
    return response_text
    #
    # return response_text

def normalize_input(cv2_im_rgb,input_details,normalize):
    if normalize == 2:
        cv2_im_rgb = (cv2_im_rgb-127.5)/127.5
    elif normalize == 1:
        cv2_im_rgb = cv2_im_rgb/255.0
    if input_details[0]['dtype']==np.float32:
        cv2_im_rgb = np.expand_dims(cv2_im_rgb,0).astype(input_details[0]['dtype'])
    elif input_details[0]['dtype']==np.uint8:
        cv2_im_rgb = quantize(input_details[0], cv2_im_rgb)
    return cv2_im_rgb

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']
    return (data/a + b).astype(dtype).reshape(shape)

def dequantize(detail, data):
    a, b = detail['quantization']
    return (data - b)*a

def load_model(camera_info):
    default_model=camera_info["model_path"]
    default_label=camera_info["label_path"]
    inference_settings = camera_info["inference_settings"]

    with open(default_label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        label = dict((int(k), v) for k, v in pairs)
    print(f"Inference with {default_label} and {default_model}")
    if inference_settings == 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(load_interpreter, default_model)
            interpreter,pid,growth = future.result()
            camera_info["load"] = growth
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(interpreter_allocate, interpreter)
            interpreter,pid,growth = future.result()
            camera_info["allocate"] = growth
    elif inference_settings == 1:
        anchors = load_anchors(os.path.dirname(default_model))
        info = load_in_out_info(os.path.dirname(default_model))
        interpreter = TrtModel(default_model,anchors,camera_info,order_outputs(info['outputs']))
    elif inference_settings == 2:
        anchors = load_anchors(os.path.dirname(default_model))
        network_info = load_in_out_info(os.path.dirname(default_model))
        network_info['outputs'] = order_outputs(network_info['outputs'])
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(tf_flow.load_model, default_model,network_info['inputs'],network_info['outputs'],camera_info['image_shape'])
            result,pid,growth = future.result()
        interpreter = tf_flow.TF_TRT_model(result[0],result[1],result[2],camera_info['normalize'],camera_info['image_shape'])
        camera_info["load"] = growth
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(tf_flow.allocate, interpreter.sess,interpreter.input,interpreter.output)
            _,pid,growth = future.result()
            camera_info["allocate"] = growth
    return interpreter,label

def decode_bbox(_rel_codes, _anchors,scale_factor=True):
    anchors = np.transpose(_anchors)
    ha = anchors[2]
    wa = anchors[3]
    ycenter_a = anchors[0]
    xcenter_a = anchors[1]

    rel_codes = np.transpose(_rel_codes)
    ty = (rel_codes[0]+rel_codes[2])/2
    tx = (rel_codes[1]+rel_codes[3])/2
    th = rel_codes[2]-rel_codes[0]
    tw = rel_codes[3]-rel_codes[1]

    if scale_factor:
        ty /= 10.0
        tx /= 10.0
        th /= 5.0
        tw /= 5.0

    ycenter = ty * ha + ycenter_a
    xcenter = tx * wa + xcenter_a
    w = np.exp(tw) * wa
    h = np.exp(th) * ha
    ymin = np.reshape(ycenter - h / 2.,(-1,1))
    xmin = np.reshape(xcenter - w / 2.,(-1,1))
    ymax = np.reshape(ycenter + h / 2.,(-1,1))
    xmax = np.reshape(xcenter + w / 2.,(-1,1))
    return np.concatenate([xmin, ymin, xmax, ymax], axis=-1)

def post_process(output,label,anchors,conf,keep_top_k=-1):
    decoded_boxes = decode_bbox(output[0],anchors)
    bbox_max_score_classes = np.argmax(output[1], axis=-1)
    index = np.where(bbox_max_score_classes>0)[0]
    if len(index) == 0:
        return [[],[],[]], False
    bbox_max_scores = np.max(output[0], axis=-1)[index]
    y_bboxes = decoded_boxes[index]
    bbox_max_classes = bbox_max_score_classes[index]
    keep_ids = nms(y_bboxes, bbox_max_scores,conf,keep_top_k=keep_top_k)
    if len(keep_ids) > 0:
        return [y_bboxes[keep_ids].tolist(),[label[i-1] for i in bbox_max_classes[keep_ids]],bbox_max_scores[keep_ids].tolist()], True
    else:
        return [[],[],[]], False


def nms(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    '''
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param bboxes: numpy array of 2D, [num_bboxes, 4]
    :param confidences: numpy array of 1D. [num_bboxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    '''
    if len(bboxes) == 0: return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)

    # if the number of final bboxes is less than keep_top_k, we need to pad it.
    # TODO
    return conf_keep_idx[pick]

@ThreadidManagement
def load_interpreter(default_model,usb=False):
    if isCoral():
        if usb==True:
            interpreter = tflite.Interpreter(model_path=default_model,experimental_delegates=[load_delegate('libedgetpu.so.1.0',{"device": "usb"})])
        else:
            interpreter = tflite.Interpreter(model_path=default_model,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = tflite.Interpreter(model_path=default_model)

    return interpreter

@ThreadidManagement
def interpreter_allocate(interpreter):
    interpreter.allocate_tensors()
    return interpreter

def load_anchors(parent_folder):
    for p in os.listdir(parent_folder):
        if p.endswith('npy'):
            path = os.path.join(parent_folder,p)
            break
    with open(path, 'rb') as f:
        anchors = np.load(f)
    return anchors

# def cuda_init():
#     cuda.init()
#     device = cuda.Device(0)
#     ctx = device.make_context()
#     return ctx,cuda

@inject_server_info
async def listen_others(stream_channels,stop_event,share_outputs,**kwargs):
    print("Listening...")
    aior = await aioredis.create_redis(f"redis://{kwargs['server']}:{kwargs['port']}",password=kwargs['AUTH_KEY'])
    while(True):
        if stop_event.is_set():
            break
        for stream in stream_channels:
            if stream not in share_outputs.keys():
                share_outputs[stream] = []
            try:
                messages = await aior.xrevrange(stream, count=1)
                if len(messages) > 0:
                    latest_id = messages[0][0].decode("utf-8").split("-")[0]
                    payload = {k.decode("utf-8"):v.decode("utf-8") if k.decode("utf-8") != "data" else v for k, v in messages[0][1].items()}

                    if int(payload["flag"]) == 1:
                        share_outputs[stream].append([json.loads(payload["outputs"]),int(latest_id),base64.b64encode(payload["data"]).decode('utf-8')])
                    # share_outputs[stream].append([payload[0],int(latest_id)])
            except Exception as e:
                print(e)
            if len(share_outputs[stream]) > 10:
                _ = share_outputs[stream].pop(0)
    aior.close()
    share_outputs = {}

def send_back_results(r,channel,current_image,matches,relation_bb_to_img,publisher_output,mapped_outputs_json):
    print("sendback")
    image = cv2.resize(current_image,(600,600))
    ret, encoded_img = cv2.imencode('.jpg', image)
    encoded_img = base64.b64encode(encoded_img).decode('utf-8')
    relation_bb_to_img["publisher_output"] = publisher_output
    # sss = time.time()"
    r.xadd("sendback",{"camera":channel,"current_img": encoded_img,"matches":json.dumps(matches),"mapped_outputs": json.dumps(mapped_outputs_json), "attr": json.dumps(relation_bb_to_img)},maxlen=10000)
