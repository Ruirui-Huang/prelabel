import cv2
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
from threading import Lock
from ..inference import Decoder
from .nms import non_max_suppression

class BackProcess:
    def __init__(self, args, max_workers=20, disable_pbar=True):
        '''
        Args:
            args (dict): 模型参数
            func_show (func): 可视化函数
        '''
        self.max_workers = max_workers
        self.disable_pbar = disable_pbar
        self.task_type = args['Task_type']
        if self.task_type == 'od':
            self.score_thr = args["Score_thr"]
            self.box_thr = args["Box_thr"]
            self.model_type = args["Model_type"]
            self.anchors = args["Anchors"]
            self.decoder = Decoder(self.model_type, True)
            self.classes = args['Class_show']['classes']
            self.is_show = args['Class_show']['is_show']
            self.num_labels = args["Num_classes"]

        self.result_dict = {}
        self.mutex = Lock()

    def run(self, data):
        features = data['features']
        path_img = data['path_img']
        img_size = data['img_size']
        scale_factor = data['scale_factor']
        padding_list = data['padding_list']

        # OD
        if self.task_type == 'od':
            # decode 需要加锁, decoder不是线程安全的
            with self.mutex:
                decoder_outputs = self.decoder(
                    feats=features,
                    conf_thres=self.score_thr,
                    num_labels=self.num_labels,
                    anchors=self.anchors
                )

            # nms 需要加锁，cv2.dnn.NMSBoxes不是线程安全的
            with self.mutex:
                bboxes, scores, llabels = non_max_suppression(
                    *decoder_outputs,
                    self.score_thr,
                    self.box_thr,
                )

            H, W = img_size
            ratio_h, ratio_w = scale_factor
            pre_label = []
            for box, label in zip(bboxes, llabels):
                if not self.is_show[label]: continue
                cls = self.classes[label]

                # 按数据预处理方式，将检测结果逆向转换到原图坐标系
                x0, y0, x1, y1 = box - np.array([padding_list[2], padding_list[0], padding_list[2], padding_list[0]])
                x0 = math.floor(min(max(x0 / ratio_w, 1), W - 1))
                y0 = math.floor(min(max(y0 / ratio_h, 1), H - 1))
                x1 = math.ceil(min(max(x1 / ratio_w, 1), W - 1))
                y1 = math.ceil(min(max(y1 / ratio_h, 1), H - 1))
                pre_label.append([x0, y0, x1, y1, cls])

        # OS
        elif self.task_type == 'os':
            H, W = img_size
            feature = features[0]
            feature = feature.transpose(1, 2, 0)  # W, H, C
            feature = cv2.resize(feature, [W, H]) # H, W, C 
            pre_label = np.argmax(feature, 2)     # H, W

        # 合并结果 需要加锁
        with self.mutex:
            if path_img in self.result_dict:
                self.result_dict[path_img].extend(pre_label)
            else:
                self.result_dict[path_img] = pre_label
    
    def forward(self, inputs):
        '''
        Args:
            inputs (queue.Queue): 推理后的batch结果
        Returns:
            result_dict (dict): 原始图片路径对应标签 {'path_img': prelabel}
        '''
        futures_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while not inputs.empty():
                input = inputs.get()

                # 拆分batch
                batch_size = len(input['path_imgs'])
                for i in range(batch_size):
                    path_img = input['path_imgs'][i]
                    # padding
                    if path_img is None: continue
                    img = input['imgs'][i]
                    img = img[np.newaxis, :]
                    img_size = input['img_sizes'][i]
                    scale_factor = input['scale_factors'][i]
                    padding_list = input['padding_lists'][i]
                    feat = []
                    for feature in input['features']:
                        feature_i = feature[i]
                        if self.task_type == 'od':
                            feature_i = feature_i[np.newaxis, :]
                        feat.append(feature_i)
                    input_dict = {
                        'path_img':path_img,
                        'img':img,
                        'img_size':img_size,
                        'scale_factor':scale_factor, 'padding_list':padding_list, 
                        'features':feat
                    }

                    futures_list.append(executor.submit(self.run, input_dict))
            
            # 线程池join
            for future in tqdm(as_completed(futures_list), total=len(futures_list), disable=self.disable_pbar):
                pass

        return self.result_dict

