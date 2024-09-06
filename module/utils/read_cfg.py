# -*- coding: utf-8 -*-
import json, copy
import os.path as osp
import numpy as np
    
class Npencoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_cfg(prelabeling_map, args):
    """模型库解析拆分一级OD和二级OD
    Args:
        prelabeling_map (dict): 模型库。以model名称作为键值，存储model的详细信息
        args (dict): 需要预标的目标类别信息
    """
    ModelType = [
        'yolov5', 
        'yolov6', 
        'yolov7',
        'yolov8'
        'yolox', 
        'rtmdet',
        'ppyoloe', 
        'ppyoloep'
    ]
    classes_parent = set()
    parent, child = {}, {}
    for cls, is_used in args.items():
        for model_name, value in prelabeling_map.items():
            if cls not in value["Used_classes"] or not is_used: continue
            num_classes = value["Num_classes"]
            assert value["Model_type"] in ModelType  and \
            value["Num_classes"] > 0 and \
            value["Score_thr"] > 0 and \
            value["Box_thr"] > 0 and \
            (value["Anchors"] == None or len(value["Anchors"]) == 3) and \
            len(value["Used_classes"]) <= num_classes and \
            len(value["Class_index"]) <= num_classes and \
            len(value["Class_index"]) == len(value["Used_classes"]) and \
            max(value["Class_index"]) < num_classes and \
            (value["Parent"] == None or len(value["Parent"]) == len(value["Used_classes"])), print("请检查config配置！")

            value["Path_model"] = osp.join("./model_zoo", model_name + "." + value["Weight_type"])
            if model_name not in parent.keys():
                child[model_name] = copy.copy(value)
                parent[model_name] = copy.copy(value)

            if "Class_show" not in child[model_name].keys():
                parent[model_name]["Class_show"] = {
                    "classes": [f"obj{i}" for i in range(num_classes)], 
                    "is_show": [0]*num_classes,
                    "exist_child": [0]*num_classes,
                }
                child[model_name]["Class_show"] = {
                    "classes": [f"obj{i}" for i in range(num_classes)], 
                    "is_show": [0]*num_classes,
                    "exist_child": [0]*num_classes,
                }

            index = value["Class_index"][value["Used_classes"].index(cls)]
            if value["Parent"] == None: 
                parent[model_name]["Class_show"]["classes"][index] = cls
                parent[model_name]["Class_show"]["is_show"][index] = 1
            else:
                if value["Parent"][index] == None:
                    parent[model_name]["Class_show"]["classes"][index] = cls
                    parent[model_name]["Class_show"]["is_show"][index] = 1
                else:
                    child[model_name]["Class_show"]["classes"][index] = cls
                    child[model_name]["Class_show"]["is_show"][index] = 1
                    classes_parent.add(value["Parent"][index])


    parent_map, child_map = [], []
    for _, map in parent.items():
        if not sum(map["Class_show"]["is_show"]): continue
        map["Class_show"]["exist_child"] = [int(i in classes_parent) for i in map["Class_show"]["classes"]]
        parent_map.append(map)

    for _, map in child.items():
        if not sum(map["Class_show"]["is_show"]): continue
        child_map.append(map)

    return parent_map, child_map