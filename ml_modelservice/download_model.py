# -*- coding: utf-8 -*-
import json
import logging
import os
import snappy
from zipfile import ZipFile

import requests

_author_ = 'luwt'
_date_ = '2020/1/10 17:06'


logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
#这里面指定要解压的目录。一般是model和columns,即模型本身文件目录和列信息目录。
MODULE_DIR_PREFIX = "model/model"
COLUMNS_DIR_PREFIX = "model/columns"
# 目前已知snappy压缩格式可能会产生，做一次过滤，保证模型文件能正常读取
SNAPPY_COMPRESS_SUFFIX = ".snappy"


def special_uncompress(file, fmt=SNAPPY_COMPRESS_SUFFIX):
    """
    :param file: 需要解压的文件路径
    :param fmt: 文件的压缩格式，指定文件的后缀
    """
    if fmt == SNAPPY_COMPRESS_SUFFIX:
        # 由于压缩是hdfs上压缩的，所以要调用hadoop_stream
        with open(file, 'rb')as fi, open(file.rstrip(SNAPPY_COMPRESS_SUFFIX), 'wb')as fo:
            snappy.hadoop_stream_decompress(fi, fo)
        os.remove(file)


def uncompress(zip_path, unzip_path):
    with ZipFile(zip_path, 'r')as f:
        #判断是否是pmml模型。压缩包结构是model/xxx.pmml
        tag = False
        for child in f.namelist():#其实这个namelist就包含两个元素[model,model/xxx.pmml]
            if child.endswith(".pmml"):
                tag = True
                logging.info("是pmml模型")
                break
        if tag == True:
            f.extractall(unzip_path)
        else:
            # 需要的目录：/model/{model,columns}
            zips = list(filter(lambda x: (
                    x.startswith(MODULE_DIR_PREFIX) or x.startswith(COLUMNS_DIR_PREFIX)
            ), f.namelist()))
            logging.info(f"The zip of list will be extracted! => {zips}")
            f.extractall(unzip_path, members=zips)
            # 解压后的路径，筛选出以.snappy结尾的压缩文件，然后再次单独解压
            compress_files = list(filter(lambda x: x.endswith(SNAPPY_COMPRESS_SUFFIX), zips))
            if compress_files:
                # 拼接需要特殊解压的文件全路径
                compress_files = list(map(lambda x: os.path.join(unzip_path, x), compress_files))
                [special_uncompress(cp_file) for cp_file in compress_files]


def save_model(local_path, content):
    """
    保存模型文件（zip压缩包）
    :param local_path:
    :param content:
    :return:
    """
    with open(local_path, 'wb')as f:
        f.write(content)


def download_model(local_path, unzip_path):
    """
    调用远程接口下载模型
    :param local_path: 模型下载后的本地保存路径，
        linux设置在根目录下 /model.zip，下载内容为zip压缩包
    :param unzip_path: 解压路径，默认解压到根目录下，即与local_path同级目录
    """
    url = os.environ.get('XQUERY_ADDR') + '/dsModel/downloadModelForModelService'
    # 本地测试url：
    # url = 'http://api.cop.com/datasience/xquery/dsModel/downloadModelForModelService'
    params = {
        "modelId": os.environ.get('MODEL_ID'),
        "version": os.environ.get('MODEL_VERSION'),
        "guid": os.environ.get('GUID')
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        logging.info(
            f"请求下载模型接口成功：url => {url}，\n参数为：=> "
            f"{json.dumps(params, indent=4, ensure_ascii=False)}"
        )
        save_model(local_path, res.content)
        # 默认取下载的zip包的同级目录作为解压目录
        uncompress(local_path, unzip_path)
        logging.info("模型下载成功")
        os.remove(local_path)
    else:
        logging.error(f"下载模型失败，reason is => {res.text}")
        raise requests.HTTPError("下载模型失败")


# os.environ['GUID'] = '4d0458a9566d4498a16a1a2f0838a606'
# os.environ['MODEL_ID'] = 'cb86e11f-a1dc-42b0-973e-6fdb43e747e8'
# os.environ['MODEL_VERSION'] = 'V1'
# download_model("/model.zip", "/")
