# -*- coding: utf-8 -*-
import os, json
import cv2
import os.path as osp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from pycocotools import mask as mask_util

colors = [np.random.randint(0, 255, 3) for _ in range(20)]

def imshow_det(path_img, path_json, classes, save_path=None, palette=colors, font_scale=0.5, thickness=4):
    """
    检测结果可视化
    Args:
    path_img (str): 图片路径
    path_json (str): json路径
    classes (list): 目标类别
    save_path (str): 可视化结果存储路径
    palette (list): 各类别颜色
    font_scale (float): 缩放系数 0 ~ 1
    thickness (int): 线条粗细
    Returns:
    """
    image = cv2.imread(path_img)
    jsonFile = open(path_json, mode='r', encoding='utf-8')
    json_data = json.load(jsonFile)
    for obj in json_data['objects']:
        [[x0, y0], [x1, y1]] = obj['coord']
        cls = obj['class']
        idx = classes.index(cls)
        if min(y1-y0, x1-x0) < 100: thickness = 2
        ((text_width, text_height), _) = cv2.getTextSize(cls, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(image, (x0, y0), (x1, y1), tuple([int(i) for i in palette[idx]]), thickness)
        if (x1 - x0) > text_width:
            cv2.rectangle(image, (x0, y0), (x0 + text_width, y0 + int(1.3 * text_height)), (0, 0, 0), -1)
            cv2.putText(image, cls, (x0, y0 + int(text_height)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), lineType=cv2.LINE_AA)

    if not osp.exists(save_path): os.makedirs(save_path, mode=0o777)
    cv2.imwrite(osp.join(save_path, osp.split(path_img)[-1]), image)

def imshow_semantic(path_img, seg, palette=colors, save_path=None, opacity=0.8):

    """
    分割结果可视化
    Args:
    path_img (str): 图片路径
    seg (ndarray): 原图坐标系下的分割掩码结果
    palette (list): 各类别颜色
    save_path (str): 可视化结果存储路径
    opacity (float): 透明度 0 ~ 1
    """
    image = cv2.imread(path_img)
    H, W, _ = image.shape
    index_list = np.unique(seg)
    mask = np.zeros((H, W)).astype(np.uint8)
    color_seg = np.zeros([H, W, 3]).astype(np.uint8)
    for index in index_list:
        if index == 0: continue
        mask[seg == index] = index
        color_seg[seg == index] = palette[index]
    image_seg = image * (1-opacity) + color_seg[..., ::-1] * opacity

    # root = osp.join(osp.dirname(osp.dirname(path_img)), 'mask')
    # if not osp.exists(root): os.makedirs(root)
    # path_mask = osp.join(root, osp.split(path_img)[-1].replace(".jpg", ".png"))
    # cv2.imwrite(path_mask, mask)

    image_show = np.concatenate((image_seg, np.ones((H, W//50, 3), dtype=np.uint8)*255, image), axis=1)

    if not osp.exists(save_path): os.makedirs(save_path, mode=0o777)
    cv2.imwrite(osp.join(save_path, osp.split(path_img)[-1]), image_show)

def imshow_seg(path_img, path_json, classes, use_rle=True, save_path=None):
    image = cv2.imread(path_img)
    img_h, img_w, _ = image.shape
    jsonFile = open(path_json, mode='r', encoding='utf-8')
    json_data = json.load(jsonFile)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    area_max = 0
    polygon_list = []
    area_list = []
    color_list = []
    for obj in json_data["objects"]:
        if classes is None:
            # 默认前景
            cls_id = 1
        else:
            if obj['class'] in classes:
                cls_id = classes.index(obj['class']) + 1
            else:
                # 作为背景
                continue

        if use_rle:
            rle = {"size":[img_h, img_w], "counts": obj['rle']}
            maskArr = np.array(mask_util.decode(rle), dtype=np.uint8)
            maskArea = maskArr.sum()
            mask += maskArr*cls_id
            mask = np.where((mask > cls_id)&(area_max < maskArea), cls_id, mask)
            area_max = max(area_max, maskArea)
        else:
            color_list.append(cls_id)
            polygon = np.array(obj['coord'])
            polygon = np.asarray([polygon])
            polygon = polygon.astype(np.int32)
            shape = polygon.shape
            polygon = polygon.reshape(shape[0], -1, 2)
            polygon_list.append(polygon)
            area = cv2.contourArea(contour=polygon, oriented=False)
            area_list.append(area)

    if not use_rle:
        area_list = np.asarray(area_list)
        area_index = np.argsort(-area_list)
        for i in range(len(area_index)):
            c_id = area_index[i]
            cv2.fillPoly(mask, polygon_list[c_id], color=color_list[c_id])

    if mask is not None:
        imshow_semantic(path_img, mask, save_path=save_path)

# 预标注结果可视化
def imshow(info, model_configs, save_path, max_workers, use_rle=1, disable_pbar=True):
    classes = set()
    task_type = model_configs[0]['Task_type']
    for m in model_configs:
        _classes = m['Class_show']['classes']
        _is_show = m['Class_show']['is_show']

    for cls, s in zip(_classes, _is_show):
        if s: classes.add(cls)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_list = []
        for line in info:
            path_img = line['file']
            path_json = line['json']
            if task_type == 'od':
                bound_func_imshow_det = partial(imshow_det, save_path=save_path)
                futures_list.append(executor.submit(bound_func_imshow_det, path_img, path_json, list(classes)))
            elif task_type == 'os':
                bound_func_imshow_seg = partial(imshow_seg, save_path=save_path)
                futures_list.append(executor.submit(bound_func_imshow_seg, path_img, path_json, list(classes), use_rle))
                
        for future in tqdm(as_completed(futures_list), total=len(futures_list), disable=disable_pbar):
            pass