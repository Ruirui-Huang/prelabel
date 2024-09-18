# -*- coding: utf-8 -*-
# @Date    : 2024-09-18 23:51:14
# @Author  : huang_rui

import os, json, argparse
import os.path as osp
def getArgs():
    parser = argparse.ArgumentParser(description="生成filesjson文件")
    parser.add_argument('image_dir', type=str)
    parser.add_argument('json_dir', type=str)
    parser.add_argument('filesInfo', type=str)
    return parser

def generate_json(image_dir, json_dir, filesInfo):
    # 如果JSON文件夹路径不存在，则创建文件夹
    if not osp.exists(json_dir):
        os.makedirs(json_dir, mode=0o777)

    data = []

    for image in os.listdir(image_dir):
        if image.endswith('.jpg'):
            image_path = osp.join(image_dir, image)
            json_name = image.replace('.jpg', '.json')
            json_path = osp.join(json_dir, json_name)
            data.append({
                "file": image_path,
                "json": json_path
            })

    # 将数据写入新的JSON文件
    with open(filesInfo, 'w') as outfile:
        json.dump(data, outfile, indent=4)

if __name__ == '__main__': 
    parser = getArgs()
    args = parser.parse_args()
    generate_json(args.image_dir, args.json_dir, args.filesInfo)