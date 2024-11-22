"""
This module provides utilities for logging.

"""

import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import tempfile


def setup_global_logger(
    logger_name=__name__, log_file=None, log_level=logging.DEBUG, log_dir="log"
):
    """
    配置全局日志记录器，在模块加载时初始化，供全局使用。

    Args:
        logger_name (str): 日志记录器的名称，默认为当前模块的名称。
        log_file (str, optional): 日志文件的路径。如果为None，则根据当前日期创建日志文件名。默认为None。
        log_level (int, optional): 日志级别。默认为logging.DEBUG。
        log_dir (str, optional): 日志文件的目录。默认为"log"。

    Returns:
        Logger: 全局日志记录器对象。
    """
    # global logger

    # 确保日志目录存在
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as e:
            log_dir = tempfile.gettempdir()  # 使用临时目录
            print(f"Failed to create log directory: {e}. Using {log_dir} instead.")

    # 如果未指定日志文件，则按日期生成文件名
    if log_file is None:
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(log_dir, f"{logger_name}_{today}.log")

    # 设置日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 定义日志格式，添加 %(name)s 显示当前记录器名称
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    )

    # 文件输出
    file_handler = TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)

    # 避免重复添加
    if not any(isinstance(handler, logging.Handler) for handler in logger.handlers):
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# 在模块加载时初始化全局日志
logger = setup_global_logger(log_dir="logs")

# 示例使用
if __name__ == "__main__":
    logger.info("测试 INFO 日志")
    logger.warning("测试 WARNING 日志")
    logger.error("测试 ERROR 日志")
    logger.debug("测试 DEBUG 日志")
