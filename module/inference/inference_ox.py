import os
import time
import onnx
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime as ort
from queue import Queue
from threading import Thread
from ..preprocess import Preprocess
from ..backprocess import BackProcess
from ..utils import save_os_json

def inference_ox(n_gpu, m, images, shared_list, mutex, info, use_rle, max_workers, semaphores, logger, disable_pbar):
    task_type = m['Task_type']
    # 检查配置
    assert m['Weight_type'] == 'onnx', '模型类型不支持'
    assert m['Class_show'], 'Class_show为空!'
    classes = m['Class_show']['classes']
    is_show = m['Class_show']['is_show']
    assert len(classes) == len(is_show), '配置有误！'
    path_model = m['Path_model']
    assert os.path.exists(path_model), f'{path_model}不存在！'
    logger.info(f'Task_type: {task_type} - Start prelabeling model: {os.path.split(path_model)[-1]}...')
    
    # 获取batch_size, input_size
    logger.info(f'Task_type: {task_type} - Start getting batch_size and input_size...')
    load_start_time = time.time()
    model = onnx.load(m['Path_model'])
    inputs = model.graph.input
    input_info = {input.name: [] for input in inputs}
    for input in inputs:
        for dim in input.type.tensor_type.shape.dim:
            if dim.dim_value:  # 确保维度值存在
                input_info[input.name].append(dim.dim_value)
            elif dim.dim_param:  # 处理动态维度
                input_info[input.name].append(dim.dim_param)

    assert input_info, 'batch_size = channel_size = input_size = None!'
    first_input_shape = input_info[next(iter(input_info))]
    assert len(first_input_shape) >= 2, 'input_size = None!'
    batch_size = first_input_shape[0]
    channel_size = first_input_shape[1]
    input_size = first_input_shape[-2:]

    load_end_time = time.time()
    logger.info(f'Task_type: {task_type} - Finish getting batch_size:{batch_size} and input size:{input_size}...')
    logger.info(f'Task_type: {task_type} - Uses {load_end_time - load_start_time} s')

    # 预处理
    logger.info(f'Task_type: {task_type} - Start preprocessing...')
    preprocess_start_time = time.time()
    pre = Preprocess(
        batch_size=batch_size, 
        input_size=input_size, 
        channel_size=channel_size, 
        fixed_scale=m['Fixed_scale'], 
        color_space=m['Color_space'], 
        logger=logger,
        disable_pbar=disable_pbar,)
    results = pre.preprocess(images)
    preprocess_end_time = time.time()
    logger.info(f'Task_type: {task_type} - Finish Preprocessing...')
    logger.info(f'Task_type: {task_type} - Uses {preprocess_end_time - preprocess_start_time} s')

    # 模型推理
    logger.info(f'Task_type: {task_type} - Start infering...')
    infer_start_time = time.time()
    inf = Inference(n_gpu, m['Path_model'], semaphores, logger)
    feats = inf.forward(results)
    infer_end_time = time.time()
    logger.info(f'Task_type: {task_type} - Finish Infering...')
    logger.info(f'Task_type: {task_type} - Uses {infer_end_time - infer_start_time} s')

    # 预处理结果回收
    del results

    # 后处理
    logger.info(f'Task_type: {task_type} - Start backprocessing...')
    back_start_time = time.time()
    back = BackProcess(m, max_workers)
    prelabels = back.forward(feats)
    back_end_time = time.time()
    logger.info(f'Task_type: {task_type} - Finish backprocessing...')
    logger.info(f'Task_type: {task_type} - Uses {back_end_time - back_start_time} s')

    # 推理结果回收
    del feats
    
    # OD标注处理流程和OS不一样，是因为OD需要额外考虑多级标签嵌套，不考虑的话，可以整合一下
    if task_type == 'od':
        shared_list.append(prelabels)
    elif task_type == 'os':
        # 保存为dahua json
        # 使用锁保证一次只有一个进程可以读写文件
        with mutex:
            logger.info(f'Task_type: {task_type} - Start saving dahua json...')
            save_start_time = time.time()
            futures_list = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for line in info:
                    result = prelabels[line['file']]
                    future = executor.submit(
                            save_os_json,
                            line['baseId'],
                            os.path.basename(line['file']),
                            result,
                            line['json'],
                            classes,
                            is_show,
                            use_rle)
                    futures_list.append(future)
                for future in tqdm(as_completed(futures_list), total=len(futures_list), disable=disable_pbar):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Task failed: {e}")

            save_end_time = time.time()
            logger.info(f'Task_type: {task_type} - End saving dahua json...')
            logger.info(f'Task_type: {task_type} - Uses {save_end_time - save_start_time} s')


class Inference:
    def __init__(self, gpu, model, semaphores, logger, disable_pbar=True):
        '''
        Args:
            gpu (int): gpu索引
            model: 模型位置
            semaphores (list): 信号量, 保证GPU卡使用只有不超过信号量的模型正在进行推理
        '''
        self.gpu = gpu
        self.model = model
        self.semaphores = semaphores
        self.logger = logger
        self.disable_pbar = disable_pbar
    
    def infer(self, gpu_id, batches, result):
        '''
        Args:
            gpu_id (int): 分配的gpu卡索引
            batches (list[dict]): 分配到指定gpu上的数据，list of {'path_imgs':,'imgs':,}
            result (queue.Queue()): 结果输出到该列表
        '''
        self.semaphores[gpu_id].acquire()
        try:
            # 分配gpu卡
            session = ort.InferenceSession(self.model, providers=['CUDAExecutionProvider'])
            session.set_providers(['CUDAExecutionProvider'], [{'device_id': gpu_id}])
            input_name = session.get_inputs()[0].name

            # 模型推理
            for batch in tqdm(batches, desc="Number " + str(gpu_id) + " gpus infering", position=gpu_id, disable=self.disable_pbar):
                input_ortvalue = ort.OrtValue.ortvalue_from_numpy(batch['imgs'], 'cuda', gpu_id)
                features = session.run(None, {input_name: input_ortvalue})
                batch['features'] = features
                result.put(batch)
        except Exception as e:
            self.logger.error(f"Error during inference on GPU {gpu_id}: {e}")
        finally:
            self.semaphores[gpu_id].release()
    
    def forward(self, batches):
        '''
        多线程推理
        Args:
            batches(list[dict]): 预处理后的图片结果
        Returns:
            results(queue.Queue): 
        '''
        result = Queue()
        segment_size = int(len(batches) / self.gpu)

        thread_list = []
        for i in range(self.gpu):
            if i == self.gpu - 1:
                t = Thread(target=self.infer, args=(i, batches[segment_size*i:], result))
            else:
                t = Thread(target=self.infer, args=(i, batches[segment_size*i:segment_size*(i+1)], result))
            thread_list.append(t)

        for t in thread_list: t.start()
        for t in thread_list: t.join()

        return result