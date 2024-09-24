# -*- coding: utf-8 -*-
# @Date    : 2024-09-24 00:07:05
# @Author  : huang_rui

import numpy as np
import json, os
import cv2
import os.path as osp
from pycocotools import mask as mask_util

maskId = np.linspace(255, 0, 20).astype(int)
class Npencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def save_od_json(baseId, path_img, path_json, result):
    """
    预标注结果写入
    Args:
    baseId (str): 图片ID
    result (list): 预标结果，其中各元素的组成为[左上角横坐标, 左上角纵坐标, 右下角横坐标, 右下角纵坐标, 类别, 父级信息]
    path_json (str): 标注文件路径
    path_img (str): 图片路径
    """
    image = cv2.imread(path_img)
    fileName = os.path.basename(path_img)
    H, W, _ = image.shape
    output_info = {
        "baseId": baseId,
        "fileName": fileName,
        "imgSize": [W, H],
        "objects": []
    }
    obj_idx = 1
    for label in result:
        x0, y0, x1, y1, cls = label
        obj_json = {
            "class": str(cls),
            "coord": [[x0, y0], [x1, y1]],
            "id": obj_idx,
            "shape": "rect",
            "props": {}
        }
        output_info["objects"].append(obj_json)
        obj_idx += 1

    with open(path_json, 'w') as json_f:
        json.dump(output_info, json_f, indent=2, cls=Npencoder)
    json_f.close()

def save_os_json(baseId, fileName, result, path_json, classes, is_show, use_rle=True):
    H, W = result.shape
    obj_idx = 1
    if osp.exists(path_json):
        jsonFile = open(path_json, mode='r', encoding='utf-8')
        output_json_info = json.load(jsonFile)
        for obj in output_json_info["objects"]:
            obj_idx = max(obj_idx, int(obj["id"])) + 1
    else:
        output_json_info = {
        "baseId": baseId,
        "fileName": fileName,
        "imgSize": [W, H],
        "objects": []
        }
        
    for idx, cls in enumerate(classes):
        if not is_show[idx]: continue
        mask = np.zeros((H, W)).astype(np.uint8)
        mask[result == idx] = 1
        mask = mask[:, :, None]
        # RLE标注
        if use_rle:
            area = mask.sum()
            if area < 10: continue
            rle = mask_util.encode(np.array(mask, order="F"))[0]
            rle = rle["counts"].decode("utf-8")
            obj_json = {
                "props": {},
                "coord": [[0, 0], [W, H]],  # 此处可以额外计算各类别外接矩形框作为coord传入
                "class": cls,
                "area": area,
                "id": int(obj_idx),
                "rle": rle,
                "maskId": maskId[classes.index(cls)],
                "shape": "mask"
                }
            output_json_info["objects"].append(obj_json)
            obj_idx += 1
        # POLY标注
        else:
            contours, _ = cv2.findContours(mask*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                polygon = contour.reshape(-1, 2)
                if polygon.shape[0] < 3 or cv2.contourArea(contour) < 9:
                    continue
                for i in range(len(polygon)):
                    polygon[i][0] = polygon[i][0]
                    polygon[i][1] = polygon[i][1]
                    obj_json = {
                        "props": {},
                        "coord": polygon,
                        "class": cls,
                        "id": int(obj_idx),
                        "shape": "poly"
                        }
                output_json_info["objects"].append(obj_json)
                obj_idx += 1
    with open(path_json, 'w') as fw:
        json.dump(output_json_info, fw, indent=2, cls=Npencoder)
    fw.close()

