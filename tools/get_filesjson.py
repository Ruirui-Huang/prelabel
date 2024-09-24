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

if __name__ == '__main__': 
    parser = getArgs()
    args = parser.parse_args()
    if not osp.exists(json_dir):
        os.makedirs(json_dir, mode=0o777)
    with open(args.filesInfo, 'w') as outfile:
        for img in os.listdir(image_dir):
            if img.endswith('.jpg'):
                img_path = osp.join(args.image_dir, img)
                json_path = osp.join(args.json_dir, img) + ".json"
                data = {
                    "baseId": "111",
                    "file": image_path,
                    "json": json_path
                }
                json.dump(data, outfile)
                outfile.write('\n')
