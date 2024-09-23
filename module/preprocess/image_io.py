from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2


class ImageIO:
    def __init__(self, max_workers=20, disable_pbar=True):
        '''
        Args:
            path_imgs: 图片路径
            max_workers: 线程数量(经过测试20为最优线程数) 
        '''
        self.disable_pbar = disable_pbar
        self.max_workers = max_workers
    
    def read_image(self, path_imgs):
        ''' 
        多线程优化读入图片
        Returns:
            images(list[dict]): list of {'path_img':, 'img':, 'img_size':,} (img_size: [W, H])
        '''
        images = []

        futures_list = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for path_img in path_imgs:
                futures_list.append(executor.submit(self.load_image, path_img, images))
            # 线程池的join
            for future in tqdm(as_completed(futures_list), total=len(futures_list), disable=self.disable_pbar):
                pass

        return images
    
    def load_image(self, image_filename, image_list):
        image = cv2.imread(image_filename)
        image_size = image.shape[:2]
        image_list.append({'path_img':image_filename,'img':image, 'img_size': image_size})