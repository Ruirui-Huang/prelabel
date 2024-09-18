# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
from tqdm import tqdm

class Preprocessor:
    """默认数据预处理方式为强行缩放到指定大小
    fixed_scale (int): 值为0，图像直接缩放
                       值为1，图像保持原图宽高比缩放，居中对齐
                       值为2，图像保持原图宽高比缩放，底部对齐，左右居中
                       值为3，图像保持原图宽高比缩放，顶部对齐，左对齐
    color_space (str): "rgb"，"bgr"
    """
    def __init__(self, fixed_scale=0, color_space="rgb"):
        self.mean = np.array([0, 0, 0], dtype=np.float32).reshape((3, 1, 1))
        self.std = np.array([255, 255, 255], dtype=np.float32).reshape((3, 1, 1))
        self.fixed_scale = fixed_scale
        assert color_space in ["rgb", "bgr"], print("数据颜色空间仅支持rgb和bgr！\n")
        self.is_rgb = True if color_space == "rgb" else False

    def __call__(self, image, new_size):
        """数据预处理
        Args: 
            image (ndarray): 图片数据
            new_size (list): onnx输入大小 H, W
        """
        assert 0 <= self.fixed_scale <=3, print(f"缩放方式{self.fixed_scale}不支持！")
        if self.is_rgb: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        height_new, width_new = new_size
        padding_list = [0]*4
        scale_factor = [new_size[0] / height, new_size[1] / width]
        if self.fixed_scale == 0:
            image = cv2.resize(image, (width_new, height_new))
        else:
            ratio = min(scale_factor)
            no_pad_shape = (int(round(height*ratio)),int(round(width*ratio)))
            padding_h, padding_w = [height_new - no_pad_shape[0], width_new - no_pad_shape[1]]
            if new_size != no_pad_shape:
                image = cv2.resize(image, (no_pad_shape[1], no_pad_shape[0]))
            
            scale_factor = (no_pad_shape[1]/width, no_pad_shape[0]/height)

            if self.fixed_scale == 1:
                top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(round(padding_w // 2 - 0.1))
                bottom_padding = padding_h - top_padding
                right_padding = padding_w - left_padding
            elif self.fixed_scale == 2:
                bottom_padding = 0
                left_padding = int(round(padding_w // 2 - 0.1))
                top_padding = padding_h - bottom_padding
                right_padding = padding_w - left_padding
            elif self.fixed_scale == 3:
                top_padding, left_padding = 0, 0
                bottom_padding = padding_h - top_padding
                right_padding = padding_w - left_padding

            padding_list = [
                top_padding, bottom_padding, left_padding, right_padding
            ]
            if top_padding != 0 or bottom_padding !=0 or left_padding !=0 or right_padding !=0:
                image = cv2.copyMakeBorder(
                    image,
                    top_padding,
                    bottom_padding,
                    left_padding,
                    right_padding,
                    cv2.BORDER_CONSTANT,
                    value=(114, 114, 114)
                )
        image = np.ascontiguousarray(image.transpose(2, 0, 1))
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image[np.newaxis], scale_factor, padding_list


class Preprocess:
    def __init__(self, batch_size, channel_size, input_size, logger, fixed_scale=0, color_space="rgb"):
        self.fixed_scale = fixed_scale
        self.color_space = color_space
        self.batch_size = batch_size
        self.channel_size = channel_size
        self.input_size = input_size
        self.preprocessor = Preprocessor(self.fixed_scale, self.color_space)
        self.logger = logger

    def preprocess(self, images):
        '''
        图片预处理支持批量处理(多batch)
        Args:
            images(list[dict]): 读入模块返回的图片 list of {'path_img':, 'img':,}
        Returns:
            result(list[dict]): 预处理后的数据，如果为多batch，则将batch_size张图片整合到一个batch, list of {'path_imgs':, 'imgs':,}, 其中的path_imgs和imgs是整合后的图片
            例如：batch_size为5: shape为[5, 3, 640, 640]
        '''
        results = []
        # 图片预处理
        # Todo: 现在默认batch_size必须整除图片数量，添加功能使得batch_size可以为任意值
        self.logger.info("Start (Preprocess) preprocess...")
        preprocess_start_time = time.time()
        preprocessed_img_list = []
        for image in tqdm(images):
            preprocessed_img, scale_factor, padding_list = self.preprocessor(image['img'], self.input_size)
            preprocessed_img_list.append({
                'path_img':image['path_img'],
                'img':preprocessed_img, 
                'img_size':image['img_size'],
                'scale_factor':scale_factor, 
                'padding_list':padding_list}
            )
        preprocess_end_time = time.time()
        self.logger.info("Finish (Preprocess) preprocess...")
        self.logger.debug(f'Use {preprocess_end_time - preprocess_start_time} s')

        # 将多张图片合并为一个batch
        self.logger.info("Start (Preprocess) concat...")
        concat_start_time = time.time()
        for i in tqdm(range(0, len(preprocessed_img_list), self.batch_size)):
            concat_img = preprocessed_img_list[i]['img']
            concat_path_img = [preprocessed_img_list[i]['path_img']]
            concat_img_size = [preprocessed_img_list[i]['img_size']]
            concat_scale_factor = [preprocessed_img_list[i]['scale_factor']]
            concat_padding_list = [preprocessed_img_list[i]['padding_list']]
            for j in range(1, self.batch_size):
                # padding
                if i + j >= len(preprocessed_img_list):
                    concat_img = np.concatenate((concat_img, np.zeros((1, self.channel_size, self.input_size[0], self.input_size[1]), dtype=np.float32)))
                    concat_path_img.append(None)
                    concat_img_size.append(None)
                    concat_scale_factor.append(None)
                    concat_padding_list.append(None)
                    continue
                concat_img = np.concatenate((concat_img, preprocessed_img_list[i+j]['img']), axis=0)
                concat_path_img.append(preprocessed_img_list[i+j]['path_img'])
                concat_img_size.append(preprocessed_img_list[i+j]['img_size'])
                concat_scale_factor.append(preprocessed_img_list[i+j]['scale_factor'])
                concat_padding_list.append(preprocessed_img_list[i+j]['padding_list'])
            results.append({
                'path_imgs':concat_path_img,
                'imgs':concat_img, 
                'img_sizes':concat_img_size,
                'scale_factors':concat_scale_factor, 'padding_lists':concat_padding_list}
            )
        concat_end_time = time.time()
        self.logger.info("Finish (Preprocess) concat...")
        self.logger.debug(f'Use {concat_end_time - concat_start_time} s')

        # gc
        preprocessed_img_list = None

        return results