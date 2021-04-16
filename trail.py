import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter(
        '%(asctime)s--文件名:%(filename)s--文件路径:%(pathname)s--函数名:%(funcName)s--行号:%(lineno)s--进程id:%(process)s--日志级别:%(levelname)s--日志内容:%(message)s')
console = logging.StreamHandler()
logger.addHandler(console)

logging.info("logging测试")
logger.info("logger测试")
