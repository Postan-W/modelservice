#@Author : wmingzhu
#@Email : wangmingzhu@bonc.com.cn
import json
import logging
import os
import shutil
from zipfile import ZipFile
import requests
from logging.handlers import RotatingFileHandler
from fnmatch import fnmatch

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = RotatingFileHandler("./logs/downloadmodellog.txt", maxBytes=1*1024*1024*50, backupCount=20,encoding="utf-8")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
        '%(asctime)s--文件名:%(filename)s--文件路径:%(pathname)s--函数名:%(funcName)s--行号:%(lineno)s--进程id:%(process)s--日志级别:%(levelname)s--日志内容:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

COMPRESSED_MODEL_PREFIX = "model/model"

#模型压缩文件都是model/model的结构，解压完成后都model/model下的内容都放在/models下
def uncompress(origin_zip):
    with ZipFile(origin_zip, 'r') as f:
        # 找出以model/model打头的目录
        zips = list(filter(lambda x: x.startswith(COMPRESSED_MODEL_PREFIX), f.namelist()))
        print("压缩文件内部情况是:",zips)
        #找到该目录下的模型文件，然后把它放在dockerfile中设定的/models/下
        if len(zips) <= 2:#对于onnx h5 pmml pb等成立
            for element in zips:
                if fnmatch(element, "*.*"):
                    try:
                        f.extract(element, "/models/")
                        # 因为这个提取操作保留了压缩文件中的目录结构，如/models/model/model/1.h5
                        # 所以先移动模型文件，然后再删除此目录结构
                        entire_path = os.path.join("/models/", element)
                        shutil.move(entire_path, "/models/")
                        shutil.rmtree("/models/model")#相当于把/models/model/model下的模型直接移到/models下，然后再把/models/model整个删掉
                        print("模型所在目录情况:",os.listdir("/models"))
                    except Exception as e:
                        logger.error(e)
        else:
            print("不是单个模型文件")
            f.extractall("/models/",zips)#相当于把关于模型的所有文件都放到了
            print("模型所在目录情况:",os.listdir("/models"))
            print("/models/model/model的具体情况是:",os.listdir("/models/model/model"))

def save_model(origin_zip, content):
    with open(origin_zip, 'wb') as f:
        f.write(content)


def download_model(origin_zip):
    print("当前使用的是v1.12版本")
    url = os.environ.get('XQUERY_ADDR') + '/dsModel/downloadModelForModelService'
    params = {
        "modelId": os.environ.get('MODEL_ID'),
        "version": os.environ.get('MODEL_VERSION'),
        "guid": os.environ.get('GUID')
    }
    try:
        res = requests.get(url, params=params)
    except requests.exceptions.Timeout as e:
        logger.error("模型下载请求错误:",e)

    if res.status_code == 200:
        logger.info(
            f"请求成功：url =>> {url}，\n参数为：=> "
            f"{json.dumps(params, indent=4, ensure_ascii=False)}"
        )
        save_model(origin_zip, res.content)
        # 默认取下载的zip包的同级目录作为解压目录
        uncompress(origin_zip)
        logger.info("模型下载成功")
        os.remove(origin_zip)  # 删掉压缩包，节省空间
        print("移除压缩包完成")

    else:
        logger.error(f"下载模型失败，reason is => {res.text}")
        raise requests.HTTPError("下载模型失败")


download_model('/root/model.zip')

