import numpy as np
import pickle
import os
import aioredis
from collections import defaultdict
import cv2
import sklearn
import asyncio
import json
import pickle
from deployment.utils import inject_server_info
from sklearn.linear_model import LinearRegression

@inject_server_info
async def read_stream(info,**kwargs):
    aior = await aioredis.create_redis(f"redis://{kwargs['server']}:{kwargs['port']}", password=kwargs['AUTH_KEY'])
    message = await aior.xrange(info)
    return message

async def regress_view(info,stream):
    cam1 = []
    cam0 = []
    messages = await read_stream(info)
    for m in messages:
        num_key = len(m[1].keys())
        if num_key ==2:
            v_0 = []
            v_1 = []
            for k, v in m[1].items():
                v = json.loads(v.decode("utf-8"))[0]
                if k.decode("utf-8") == "outputs":
                    v_0 = v
                if k.decode("utf-8") == "inputs":
                    v_1 = v
            if len(v_0) > 0 and len(v_1)>0 and len(v_0) == len(v_1):
                cam0.append(v_0)
                cam1.append(v_1)

    cam0 = np.concatenate(cam0,axis=0)
    cam1 = np.concatenate(cam1,axis=0)
    # tempt = np.empty_like(cam0)
    # cam = [cam0,cam1]
    # for i,c in enumerate(cam):
    #     tempt[:,0] = (c[:,0]+c[:,2])/2
    #     tempt[:,1] = (c[:,1]+c[:,3])/2
    #     tempt[:,2] = (c[:,2]-c[:,0])
    #     tempt[:,3] = (c[:,3]-c[:,1])
    #     cam[i] = tempt
    reg = LinearRegression(n_jobs=4).fit(cam1, cam0)
    # print(reg.predict(cam1[250:260]),cam0[250:260])
    filename = f'./mapping/{stream}.sav'
    pickle.dump(reg, open(filename, 'wb'))


class MatchingLogic():
    def __init__(self):
        self.matching_model = {}
        self.previous_output_sub = defaultdict(list)
        self.time_out = {}
        self.bbs = []
        self.labels = []
        self.scores = []
        self.color_pallete = [(54, 96, 244),(78, 41, 46),(139, 153, 27),(54, 29, 231),(109, 216, 197)]
        self.remove_key = []
        self.matches = {"id":{},"bb_group":{}}
        # self.matching_model["147.46.116.138:40228_1"] = pickle.load(open(f'./mapping/147.46.116.138:40228_1.sav', 'rb'))


    def predict(self,output_pub,stream,concate_list,relation_bb_to_img=None,background=None):
        bbs = np.asarray(output_pub[0])
        if relation_bb_to_img is not None:
            self.marking_bbs(relation_bb_to_img,len(concate_list[1])+len(bbs),stream,background)
        result = self.matching_model[stream].predict(bbs)

        concate_list[0] = concate_list[0]+result.tolist()
        concate_list[1] = concate_list[1]+output_pub[1]
        concate_list[2] = concate_list[2]+output_pub[2]
        concate_list[3] = concate_list[3]+output_pub[0]
        # result = self.filter(result.clip(min=0,max=1))
        # return result

    def marking_bbs(self,relation_bb_to_img,end,stream,background):
        relation_bb_to_img['camera'].append(stream)
        relation_bb_to_img['background'].append(background)
        relation_bb_to_img['index'].append(end)

    def store_matching_model(self,stream):
        # self.matching_model[stream] = pickle.load(open(f'./mapping/{stream}.sav', 'rb'))
        self.matching_model[stream] = pickle.load(open(f'./mapping/mapping.sav', 'rb'))

    def is_share_view_read(self,stream):
        if stream in self.matching_model.keys():
            return True
        else:
            # if os.path.isfile(f'./mapping/{stream}.sav'):
            if os.path.isfile(f'./mapping/mapping.sav'):
                print("Reloading Mapping")
                self.store_matching_model(stream)
                return True
            else:
                return False
    def filter(result):
        invalid = set(np.where(((result[:,0]==0)&(result[:,2]==0))|((result[:,1]==0)&(result[:,3]==0))|\
                                 ((result[:,0]==1)&(result[:,2]==1))|((result[:,1]==1)&(result[:,3]==1)))[0])
        valid = list(set(range(0,len(result)))-set(invalid))
        return result[np.asarray(valid).astype(np.int32)]

    def ios(self,bb,bbs,first = True):
        x1 = np.maximum(bb[0], bbs[:,0])
        x2 = np.minimum(bb[2], bbs[:,2])
        y1 = np.maximum(bb[1], bbs[:,1])
        y2 = np.minimum(bb[3], bbs[:,3])
        w = np.maximum(0.0, x2 - x1)
        h = np.maximum(0.0, y2 - y1)
        intersection = w * h
        # print(intersection,self.calculate_area_ratio(bb_pub.reshape(1,-1)),self.calculate_area_ratio(bbs_sub))
        if first:
            ratio = intersection / self.calculate_area_ratio(bb.reshape(1,-1))
        else:
            ratio = intersection / self.calculate_area_ratio(bbs)
        return ratio

    def regress_multiple_label_by_ios(self,output_pub,output_sub,threshold=0.4):
        if len(output_pub[0])==0:
            # self.store_state(output_sub[0],output_sub[1],output_sub[2])
            return output_pub,output_sub

        bbs_sub = np.asarray(output_sub[0].copy())
        score_sub = np.asarray(output_sub[2].copy())
        label_sub = output_sub[1].copy()

        bbs_pub = np.asarray(output_pub[0].copy())
        score_pub = np.asarray(output_pub[2].copy())
        label_pub = output_pub[1].copy()
        self.matches = {"id":{},"bb_group":{}}
        ref = 0
        order = np.argsort(score_sub)
        outputs = [[],[],[]]
        skip_box = []
        while order.size > 0:
            index = order[-1]
            order = order[:-1]
            if index in skip_box:
                continue
            bb_sub = bbs_sub[index]
            ratio = self.ios(bb_sub,bbs_pub,False)
            cluster = np.where(ratio>threshold)
            overlap_element = cluster[0].tolist()
            for i in overlap_element:
                bbs_pub[i] = self.empty_array()
            skip_box = skip_box+overlap_element
            outputs[0].append(bb_sub)
            cluster_score = (np.sum(score_pub[cluster])+score_sub[index])/(len(overlap_element)+1)
            outputs[2].append(cluster_score)
            cluster_cls = label_sub[index]+'|'+'|'.join([label_pub[elem] for elem in overlap_element])
            outputs[1].append(cluster_cls)
            self.matches["bb_group"][ref] = overlap_element
            ref = ref + 1

        rest_bb = [j for j,i in enumerate(bbs_pub) if j not in skip_box]
        skip_box = []
        potential = [[],[],[]]
        if len(rest_bb)>0:
            rest_pubs = bbs_pub[rest_bb]
            rest_score = score_pub[rest_bb]
            rest_label = [label_pub[i] for i in rest_bb]
            order = np.argsort(rest_score)
            while order.size > 0:
                index = order[-1]
                order = order[:-1]
                if index in skip_box:
                    continue
                bb = rest_pubs[index]
                ratio = self.ios(bb,rest_pubs,False)
                cluster = np.where(ratio>threshold)
                overlap_element = cluster[0].tolist()
                cluster_box = rest_pubs[cluster]
                cluster_box = [np.min(cluster_box[:,0]),np.min(cluster_box[:,1]),np.max(cluster_box[:,2]),np.max(cluster_box[:,3])]
                potential[0].append(cluster_box)
                outputs[0].append(cluster_box)
                for i in overlap_element:
                    rest_pubs[i] = self.empty_array()
                skip_box = skip_box+overlap_element
                cluster_score = (np.sum(rest_score[cluster])+rest_score[index])/(len(overlap_element)+1)
                potential[2].append(cluster_score)
                outputs[2].append(cluster_score)
                cluster_cls = label_sub[index]+'|'+'|'.join([rest_label[elem] for elem in overlap_element])
                potential[1].append(cluster_cls)
                outputs[1].append(cluster_cls)
                self.matches["bb_group"][ref] = [rest_bb[i] for i in overlap_element]
                ref = ref + 1

        self.store_state(np.asarray(outputs[0]),outputs[1],np.asarray(outputs[2]))
        return potential,outputs
        # self.store_previous_state()


    def regress_object_label_by_ios(self,output_pub,output_sub,threshold=0.4):
        bbs_sub = np.asarray(output_sub[0]).copy()
        score_sub = np.asarray(output_sub[2]).copy()
        bbs_pub = np.asarray(output_pub[0]).copy()
        label_pub = output_pub[1]
        score_pub = np.asarray(output_pub[2]).copy()
        order = np.argsort(score_pub)
        potential_pub_key = []
        potential_match_label = []
        matches = {}
        # chosen_sub_key = []
        # match_label = []
        while order.size > 0:
            index = order[-1]
            bb_pub = bbs_pub[index]
            ratio = self.ios(bb_pub,bbs_sub)
            if max(ratio) >threshold:
                inter_index = np.argmax(ratio)
                matches[inter_index] = index
                output_sub[2][inter_index] = (output_sub[2][inter_index]+score_pub[index])/2
                output_sub[1][inter_index] = output_sub[1][inter_index]+label_pub[index]
                bbs_sub[inter_index] = self.empty_array()

                # chosen_sub_key.append(inter_index)
                # match_label.append(label_pub[index])

            else:
                if score_pub[index] > 0.7:
                    potential_pub_key.append(index)
                    potential_match_label.append(label_pub[index])
                # else:
                #     _ = output_pub[1].pop(index)
            order = order[:-1]
        potential_pub_key = np.asarray(potential_pub_key).astype(np.int32)
        # chosen_sub_key = np.asarray(chosen_sub_key).astype(np.int32)
        output_pub[0] = bbs_pub[potential_pub_key]
        output_pub[1] = potential_match_label
        output_pub[2] = score_pub[potential_pub_key]
        # output_sub[0] = bbs_sub[chosen_sub_key]
        # output_sub[1] = match_label
        # output_sub[2] = np.asarray(output_sub[2])[chosen_sub_key]
        # return output_pub,output_sub
        if len(output_pub[0])==0:
            self.store_state(output_sub[0],output_sub[1],output_sub[2])
        else:
            self.store_state(np.concatenate((np.asarray(output_sub[0]),output_pub[0])),output_sub[1]+output_pub[1],np.concatenate((score_sub,output_pub[2])))
        # self.store_previous_state()
        return output_pub,output_sub,matches

    # def store_previous_state(self):
        # self.previous_output_sub.append(self.bbs)

    def store_state(self,bbs,labels,scores):
        self.bbs = np.asarray(bbs)
        self.labels = labels
        self.scores = np.asarray(scores)

    def output_to_json(self):
        return [self.bbs.tolist(),self.labels,self.scores.tolist()]

    def compare_previous_state(self,threshold=0.5):
        if len(self.bbs)>0:
            if len(self.previous_output_sub.keys()) == 0:
                for i,bb in enumerate(self.bbs):
                    self.matches['id'][i] = i
                    self.previous_output_sub[i].append(bb)
                    self.labels[i] = str(i)+"_"+self.labels[i]
                    self.time_out[i] = 0
            else:
                current_bbs = self.bbs.copy()
                self.remove_key = []
                use_bb = []
                order = np.asarray(list(self.previous_output_sub.keys()))
                self.initial_order = order.copy()
                # print("OBSERVING BBS: ", self.bbs)
                # print("OLD_BB: ", self.previous_output_sub)
                # print("ORDER",initial_order)
                while len(order) > 0:
                    key = order[-1]
                    bb = self.previous_output_sub[key][-1]
                    ratio = self.ios(bb,current_bbs)
                    if max(ratio) >threshold:
                        inter_index = int(np.argmax(ratio))
                        self.matches['id'][inter_index] = int(key)
                        # print("BB INDEX",inter_index)
                        self.labels[inter_index] = str(key)+"_"+self.labels[inter_index]+"_"+self.determine_movement(bb,self.bbs[inter_index])
                        self.previous_output_sub[key].append(self.bbs[inter_index])
                        current_bbs[inter_index] = self.empty_array()
                        use_bb.append(inter_index)
                        self.time_out[key] = 0
                        # chosen_sub_key.append(inter_index)
                        # match_label.append(label_pub[index])
                    else:
                        self.time_out[key] = self.time_out[key] + 1
                        if not self.at_edge(bb) and self.time_out[key]>5:
                            self.remove_key.append(key)
                        else:
                            if self.time_out[key]>3:
                                self.remove_key.append(key)

                    order = order[:-1]
                # print("THE KEY USE;",use_bb)
                for i,bb in enumerate(self.bbs):
                    if i not in use_bb:
                        key = max(self.previous_output_sub.keys())+1
                        self.matches['id'][i] = int(key)
                        self.previous_output_sub[key].append(bb)
                        self.labels[i] = str(key)+"_"+self.labels[i]
                        self.time_out[key] = 0

    def remove(self):
        _ = [self.previous_output_sub.pop(k) for k in self.remove_key]
        _ = [self.time_out.pop(k,None) for k in self.remove_key]

    def calculate_area_ratio(self,bbarray):
        w = bbarray[:,2]-bbarray[:,0]
        h = bbarray[:,3]-bbarray[:,1]
        # return [area,area,area]
        return w*h

    def center_bb(self,bbarray,img_orig_shape):
        center_x = ((bbarray[:,2]+bbarray[:,0])/2)*img_orig_shape[1]
        center_y = ((bbarray[:,3]+bbarray[:,1])/2)*img_orig_shape[0]
        # return [[x,y],[x,y],[x,y]]
        return (np.concatenate((center_x.reshape(1,-1),center_y.reshape(1,-1))).T).astype(np.int32)

    def empty_array(self):
        return np.asarray([-1000,-1000,0,0])

    def draw_tracks(self,image,img_orig_shape):
        for key in self.previous_output_sub.keys():
            track = np.asarray(self.previous_output_sub[key])
            track = self.center_bb(track,img_orig_shape)
            color = self.color_pallete[key%len(self.color_pallete)]
            image = cv2.polylines(image, [track],
                      False, color , 2)
        return image

    def at_edge(self,bb):
        padvalue = 0.001
        center = [(bb[2]+bb[0])/2,(bb[3]+bb[1])/2]
        w = bb[2]-bb[0]
        h = bb[3]-bb[1]
        if center[0]+w >=(1-padvalue) or center[0]-w <=padvalue or center[1]+h>=(1-padvalue) or center[1]-h<=padvalue:
            return True
        return False

    def determine_movement(self,bb1,bb2):
        vibrant_value = 0.02
        center1 = [(bb1[2]+bb1[0])/2,(bb1[3]+bb1[1])/2]
        center2 = [(bb2[2]+bb2[0])/2,(bb2[3]+bb2[1])/2]
        if abs(center2[1]-center1[1]) <=vibrant_value and abs(center2[0]-center1[0]) <=vibrant_value:
            return "stay"
        else:
            if abs(center2[1]-center1[1]) <=vibrant_value:
                return "left" if (center2[0]-center1[0]) <=-vibrant_value else "right"
            if abs(center2[0]-center1[0]) <=vibrant_value:
                return "up" if (center2[1]-center1[1]) <=-vibrant_value else "down"
            else:
                if (center2[1]-center1[1]) >= vibrant_value:
                    return "left-down" if (center2[0]-center1[0]) <=-vibrant_value else "right-down"
                else:
                    return "left-up" if (center2[0]-center1[0]) <=-vibrant_value else "right-up"

    def in_out(self,key):
        try:
            state = self.determine_movement(self.previous_output_sub[key][-1],self.previous_output_sub[key][0])
            if "up" in state:
                return f"{key}_get_in", True
            if "down" in state:
                return f"{key}_get_out", True
            else:
                return 0 , False
        except:
            return 0, False
