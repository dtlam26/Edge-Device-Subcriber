import cv2
import os
import numpy as np
import time
import json

def pad2square(image):
    h,w,c = image.shape
    if w>h:
        return zero_pad_img(image,[int((w-h)/2),int((w-h)/2),0,0])
    elif h>w:
        return zero_pad_img(image,[0,0,int((h-w)/2),int((h-w)/2)])
    else:
        return image

def unpad_from_square(xyxy,w,h):
    if h>w:
        xyxy[:,:4:2] = (xyxy[:,:4:2]-0.5)*h/w+0.5
    elif w>h:
        xyxy[:,1:4:2] = (xyxy[:,1:4:2]-0.5)*w/h+0.5
    return xyxy

def zero_pad_img(image,tblr=[0,0,0,0]):
    pad_0 = [0,0,0]
    return cv2.copyMakeBorder(image.copy(),tblr[0],tblr[1],tblr[2],tblr[3],cv2.BORDER_CONSTANT,value=pad_0)


def load_in_out_info(folder):
    with open(os.path.join(folder,'model.json')) as json_file:
        data = json.load(json_file)
    return data

def order_outputs(outputs):
    """outputs should be class score then box"""
    ordered_outputs = outputs.copy()
    for i,o in enumerate(outputs):
        if 'box' in o:
            ordered_outputs[0] = len(outputs)-i-1
        elif 'class' in o:
            ordered_outputs[1] = len(outputs)-i-1
        elif 'anchor' in o:
            ordered_outputs[2] = len(outputs)-i-1
    print(outputs,ordered_outputs)
    return ordered_outputs

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False, multi_label=False,max_det=300,
                       width=1920,height=1080,object_score=True,bb_format_code=1):
    """Runs Non-Maximum Suppression (NMS) on inference results
         prediction: as an array
         if             object_ore [n_anchors x [bboxes[4]]+[object_score[1]]+[multilass_score[num_class]]
         else             object_ore [n_anchors x [bboxes[4]]+[multilass_score_with_background[num_class+1]]
         bb_format_code: 0 - center width height  1 - 4 corners
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Settings
    min_wh, max_wh = 0.001, 1  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    nc = prediction.shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    t = time.time()
    output = np.zeros((0, 6))

    # Compute conf
    if object_score:
        prediction[:, 5:] *= prediction[:, 4:5]  # conf = obj_conf * cls_conf
        multi_score_index = 5
        xc = np.where(prediction[:, 4] > conf_thres)  # candidates
        prediction = prediction[xc]  # confidence
    else:
        multi_score_index = 4

    if bb_format_code==0:
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        whc = np.where((prediction[:, 2:4] > min_wh) & (prediction[:, 2:4] < max_wh)) # width-height
        prediction = prediction[whc[0]]
        box = xywh2xyxy(prediction[:, :4])


    elif bb_format_code==1:
        box = prediction[:, :4]
    else:
        raise ValueError(f'Not support this bounding box format code: {bb_format_code}')

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = np.where(prediction[:, multi_score_index:] > conf_thres)

        if object_score:
            xxyxy, conf, cls = box[i], prediction[i, j + multi_score_index].reshape(-1,1), j.astype(np.float32).reshape(-1,1)
        else:
            c = np.where(j>0)
            xxyxy, conf, cls = box[i[c]], prediction[i[c], j[c] + multi_score_index].reshape(-1,1), j[c].astype(np.float32).reshape(-1,1)

    else:  # best class only
        j = np.argmax(prediction[:, multi_score_index:],1)
        max_conf = np.max(prediction[:,multi_score_index:],1)
        if object_score:
            i = np.where(max_conf > conf_thres)
        else:
            i = np.where((max_conf > conf_thres) & (j>0))
        xyxy = box[i]
        cls = j[i].reshape(-1,1).astype(np.float32)
        conf = max_conf[i].reshape(-1,1)


    # Check shape
    n = xyxy.shape[0]  # number of boxes

    if not n:  # no boxes
        return output, []
    else:
        # print(xyxy.shape,cls.shape,cls.shape)
        x = np.concatenate([xyxy,conf,cls],1)
        x = x[np.argsort(x[:, 4])][:max_nms,:]  # sort by confidence
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = single_class_non_max_suppression(boxes, scores, iou_thresh=iou_thres)  # NMS

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

#         if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#             # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#             iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#             weights = iou * scores[None]  # box weights
#             x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#             if redundant:
#                 i = i[iou.sum(1) > 1]  # require redundancy

        output = x[i]
        # print(time.time()-t)
        return output, i

def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.25, iou_thresh=0.45, keep_top_k=-1,skip_sort=True):
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
    if skip_sort:
        idxs = np.arange(len(confidences))
    else:
        conf_keep_idx = np.where(confidences > conf_thresh)[0]
        bboxes = bboxes[conf_keep_idx]
        confidences = confidences[conf_keep_idx]
        idxs = np.argsort(confidences)


    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)


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
    return np.asarray(pick)
