import os, sys
import time
import psutil
import pynvml
import math
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Manager, Lock, Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from module.preprocess import ImageIO
from module.inference import inference_ox
from module.utils import ReadConfig, setup_logging, imshow

# 标签融合
def merge_labels(df, prelabels):
    p_bar = tqdm(prelabels.items())
    p_bar.set_description('merge labels')
    for path_img, labels in p_bar:
        if path_img not in df['path_img'].values:
            df.loc[len(df)] = [path_img, labels]
        else:
            index = df['path_img'][df['path_img'].values == path_img].index
            df.loc[index, 'labels'].values[0].extend(labels)

def generate_dahua_json(info, df, max_workers):
    futures_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for line in info:
            baseId = line['baseId']
            path_img = line['file']
            path_json = line['json']
            index = df['path_img'][df['path_img'].values == path_img].index
            labels = df['labels'].values[index]
            if not len(labels): continue
            futures_list.append(executor.submit(save_od_json, baseId, path_img, path_json, labels[0]))
        
        for future in tqdm(as_completed(futures_list), total=len(futures_list)):
            pass

def main():
    program_start_time = time.time()
    # 参数解析
    cfg = ReadConfig()
    info, inputInfo, model_configs = cfg.parse_all()
    path_imgs = [a['file'] for a in info]
    pynvml.nvmlInit()
    n_gpu = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    draw_path = inputInfo['args'].get('draw_path', None)
    use_rle = inputInfo['args'].get('use_rle', 1)
    load_size_per_gpu = min(inputInfo['args'].get('load_size', 5000), len(path_imgs))
    max_workers = inputInfo['args'].get('max_workers', 20)
    model_num_per_gpu = inputInfo['args'].get('model_num_per_gpu', 1)
    log_level = inputInfo['args'].get('log_level', 20)
    task_type = inputInfo['args']['task_type']
    load_size = load_size_per_gpu*n_gpu
    # 日志
    logger = setup_logging(level=log_level)
    logger.info(f'Task_type: {task_type}...')
    segment_num = math.ceil(len(path_imgs) / load_size)
    images = None
    for i in range(segment_num):
        # 检查内存使用情况
        mem = psutil.virtual_memory()
        if mem.percent > 80:
            logger.error("Memory usage is above the threshold: 80%")
            exit(1)
        io_start_time = time.time()
        if i == segment_num - 1:
            logger.info(f'Task_type: {task_type} - Start reading images [{i*load_size}:{len(path_imgs)}]...')
            io = ImageIO(path_imgs[i*load_size:], max_workers)
            images = io.read_image()
            _info = info[i*load_size:]
        else:
            logger.info(f'Task_type: {task_type} - Start reading images [{i*load_size}:{(i+1)*load_size}]...')
            io = ImageIO(path_imgs[i*load_size:(i+1)*load_size], max_workers)
            images = io.read_image()
            _info = info[i*load_size:(i+1)*load_size]
        io_end_time = time.time()
        logger.info(f'Task_type: {task_type} - Finish reading images...')
        logger.debug(f'Task_type: {task_type} - Uses {io_end_time - io_start_time} s')

        shared_list = Manager().list()
        df = pd.DataFrame(columns=['path_img', 'labels'])
        process_list = []
        mutex = Lock()
        semaphores = [Semaphore(model_num_per_gpu) for _ in range(n_gpu)]
        for m in model_configs:
            p = Process(target=inference_ox, args=(n_gpu, m, images, shared_list, mutex, _info, use_rle, max_workers, semaphores, logger,))
            process_list.append(p)

        [p.start() for p in process_list]
        [p.join() for p in process_list]

        # if task_type == 'od':
        #     # 标签融合
        #     logger.info(f'Task_type: {task_type} - Start merging labels...')
        #     merge_label_start_time = time.time()
        #     for prelabel in shared_list:
        #         merge_labels(df, prelabel)
        #     merge_label_end_time = time.time()
        #     logger.info(f'Task_type: {task_type} - Finish merging labels...')
        #     logger.debug(f'Task_type: {task_type} - Use {merge_label_end_time - merge_label_start_time} s')

        #     # 生成dahua json
        #     logger.info(f"Task_type: {task_type} - Start generating dahua jsons...")
        #     generate_start_time = time.time()
        #     generate_dahua_json(_info, df, max_workers)
        #     generate_end_time = time.time()
        #     logger.info(f'Task_type: {task_type} - Finish generating dahua jsons...')
        #     logger.debug(f'Task_type: {task_type} - Use {generate_end_time - generate_start_time} s')

    program_end_time = time.time()
    logger.debug(f'Task_type: {task_type} - program use {program_end_time - program_start_time} s')

    # 结果可视化
    if draw_path is not None:
        logger.info(f'Task_type: {task_type} - Start drawing...')
        imshow(info, model_configs, draw_path, max_workers)
        logger.info(f'Task_type: {task_type} - Finish drawing...')

if __name__ == '__main__':
    main()