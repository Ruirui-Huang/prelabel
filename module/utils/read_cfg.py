# -*- coding: utf-8 -*-
import json, copy, argparse
import os.path as osp
import numpy as np
from mmengine import Config

class ReadConfig:
    def __init__(self):
        parser = self.getArgs()
        self.args = parser.parse_args()
        self.prelabeling_map_path = './configs/config.py'

    def getArgs(self):
        """
        巨灵平台获取该方法参数的入口
        """
        parser = argparse.ArgumentParser(description="行业预标注工具！")
        parser.add_argument('--token', type=str, required=True)
        parser.add_argument('--jsonFile', type=str, required=True)
        parser.add_argument('--filesInfo', type=str, required=True)
        return parser

    def parse_platform_config(self):
        '''
        解析平台参数
        Returns:
            info (list[dict]): 图片、json路径
            inputInfo (dict): 平台参数以及需要预标的目标类别信息
        '''
        fileInfo = open(self.args.filesInfo, 'r', encoding='utf-8')
        jsonFile = open(self.args.jsonFile, mode='r', encoding='utf-8')
        info_lines = fileInfo.readlines()
        inputInfo = json.load(jsonFile)

        info = []
        for i in info_lines:
            jsonInfo = json.loads(i)
            if jsonInfo['file'].endswith('.jpg'):
                # workflow上预标注json存储要求，本地运行可以注释掉
                jsonInfo['json'] = jsonInfo['json'].replace(inputInfo["markDataInPath"], inputInfo["markDataOutPath"])
                info.append(jsonInfo)

        jsonFile.close()
        fileInfo.close()
        return info, inputInfo

    def parse_model_config(self, model_args, task_type):
        '''
        解析模型参数
        Args:
            model_args (dict): 需要预标的目标类别信息
            task_type (str): 任务类型, 'od' 或 'os' 或 'oc'
        Returns:
            model_configs (list[dict]): 预标的目标类别需要用到的模型参数
        '''
        cfg = Config.fromfile(self.prelabeling_map_path)
        prelabeling_map = cfg.get('prelabeling_map', None)
        assert prelabeling_map, print('检查prelabeling_map配置')

        ModelType = [
        'yolov5', 
        'yolov6', 
        'yolov7',
        'yolov8',
        'yolox', 
        'rtmdet',
        'ppyoloe', 
        'ppyoloep'
        ]
        model_config = {}
        for cls, is_used in model_args.items():
            for model_name, value in prelabeling_map.items():
                # 检查prelabeling_map
                assert value['Num_classes'] > 0, print(f'模型{model_name}的类别数未配置')
                num_classes = value['Num_classes']

                if "Model_type" in value.keys() and \
                    "Score_thr" in value.keys() and \
                    "Box_thr" in value.keys() and \
                    "Anchors" in value.keys():
                    assert value["Model_type"] in ModelType and \
                    value["Score_thr"] > 0 and \
                    value["Box_thr"] > 0 and \
                    (value["Anchors"] == None or len(value["Anchors"]) == 3), print("请检查config配置！")
                
                assert value["Task_type"] in ["od", "os"] and \
                len(value["Used_classes"]) <= num_classes and \
                len(value["Class_index"]) <= num_classes and \
                len(value["Class_index"]) == len(value["Used_classes"]) and \
                max(value["Class_index"]) < num_classes, print("请检查config配置！")

                if cls not in value["Used_classes"] or \
                not is_used or \
                value["Task_type"] != task_type: continue
                
                # 添加Path_model以及Class_show
                value["Path_model"] = osp.join('./model_zoo', model_name + '.' + value['Weight_type'])
                if model_name not in model_config.keys():
                    model_config[model_name] = copy.copy(value)
                
                if 'Class_show' not in model_config[model_name].keys():
                    model_config[model_name]['Class_show'] = {
                        'classes': [f'obj{i}' for i in range(num_classes)],
                        'is_show': [0] * num_classes
                    }
                
                # 目标检测类别在对应检测模型classes下的index
                index = value['Class_index'][value['Used_classes'].index(cls)]
                model_config[model_name]['Class_show']['classes'][index] = cls
                model_config[model_name]['Class_show']['is_show'][index] = 1

        model_configs = []
        for _, map in model_config.items():
            if not sum(map['Class_show']['is_show']): continue
            model_configs.append(map)
        
        return model_configs

    def parse_all(self):
        ''' 解析平台+模型参数
        Args:
            task_type (str): 任务类型, 'od' 或 'os' 或 'oc'
        Returns:
            info (list[dict]): 图片、json路径
            inputInfo (dict): 平台参数以及需要预标的目标类别信息
            model_configs (list[dict]): 预标的目标类别需要用到的模型参数
        '''
        info, inputInfo = self.parse_platform_config()
        task_type = inputInfo['args'].get('task_type', None)
        assert task_type, print('未配置task_type！')
        model_configs = self.parse_model_config(inputInfo['args'], task_type)

        return info, inputInfo, model_configs