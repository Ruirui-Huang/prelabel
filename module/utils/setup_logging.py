# log_config.py
import logging
def setup_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)

    # 创建一个日志处理器，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 创建一个日志处理器，用于输出到文件
    # file_handler = logging.FileHandler('app.log')
    # file_handler.setLevel(level)

    # 创建一个日志格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式器添加到处理器
    console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger