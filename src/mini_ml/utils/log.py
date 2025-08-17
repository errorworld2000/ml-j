# log_utils.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_LEVEL_FILE = logging.DEBUG
DEFAULT_LOG_LEVEL_CONSOLE = logging.INFO
DEFAULT_LOG_FILE_NAME = "app.log"
LOG_BACKUP_COUNT = 7


def setup_logger(
    log_dir=DEFAULT_LOG_DIR,
    logger_name=None,
    file_level=DEFAULT_LOG_LEVEL_FILE,
    console_level=DEFAULT_LOG_LEVEL_CONSOLE,
):
    """
    配置日志记录器（全局或指定名称）。
    如果 logger_name 为 None，则配置 root logger。

    Args:
        log_dir (str): 日志文件目录
        logger_name (str|None): logger 名称
        file_level (int): 文件日志级别
        console_level (int): 控制台日志级别

    Returns:
        logging.Logger: 配置好的 logger 对象
    """
    # 获取 logger
    log_obj = logging.getLogger(logger_name)
    log_obj.setLevel(logging.DEBUG)  # 总体捕获所有级别，Handler 决定输出级别

    # 防止重复添加 Handler
    if log_obj.handlers:
        return log_obj

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(
        log_dir, DEFAULT_LOG_FILE_NAME if logger_name is None else f"{logger_name}.log"
    )

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # 文件 Handler
    file_handler = TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    log_obj.addHandler(file_handler)

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    log_obj.addHandler(console_handler)

    return log_obj


global_logger = setup_logger()

if __name__ == "__main__":
    # 模块级 logger
    logger = logging.getLogger(__name__)

    logger.debug("DEBUG 日志示例")
    logger.info("INFO 日志示例")
    logger.warning("WARNING 日志示例")
    logger.error("ERROR 日志示例")
    logger.critical("CRITICAL 日志示例")
