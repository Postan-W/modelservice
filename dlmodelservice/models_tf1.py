import abc
import base64
import io
import os
import onnxruntime
import requests
import tensorflow as tf
from pypmml import Model
import tensorflow as tf
import keras
#如果CUDA不能正常使用，那就禁用
if not tf.test.is_built_with_cuda():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from local_utilities import analysis_list
import numpy as np
import operator
from PIL import Image
import logging
import json
from local_utilities import h5_resize_bs64,h5_input_shape,universal_image_process,savemodel_input_shape
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = RotatingFileHandler("./logs/models_error.txt", maxBytes=1*1024*1024*50, backupCount=20,encoding="utf-8")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
        '%(asctime)s--文件名:%(filename)s--文件路径:%(pathname)s--函数名:%(funcName)s--行号:%(lineno)s--进程id:%(process)s--日志级别:%(levelname)s--日志内容:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)
"""
抽象的模型类，需要实现为以下几个模型格式：savedmodel、onnx、pmml、h5、ckpt、pb、pth
"""

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
}


class PublicModelInterface(abc.ABC):

    @abc.abstractmethod
    def predict(self, data):
        pass

    @abc.abstractmethod
    def get_info(self):
        pass


class PMMLModel(PublicModelInterface):

    def __init__(self, modelpath,model_inputs=None):
        self.model = Model.fromfile(modelpath)
        self.info = ".pmml"

    def __del__(self):
        if self.model:
            self.model.close()

    def get_info(self):
        return self.info

    def predict(self, data):
        return self.model.predict(data)


class H5Model(PublicModelInterface):

    def __init__(self, modelpath,model_inputs=None):

        self.model = keras.models.load_model(modelpath)
        self.info = self.model.to_json()#这是模型的全部信息json格式，即str，内容较多，不易查看
        logging.info("模型基本信息:::")
        logging.info(self.info)

    def get_info(self):
        return self.info

    def predict(self, data):
        shape = h5_input_shape(self.info,logger)
        print("该模型的输入形状是:",shape)
        if data["type"] == "b64":
            #如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            print("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)
            img_b64 = universal_image_process(img_b64,shape,logger)
            result = self.model.predict(img_b64)
            # print("预测结果是:",result)
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content
                print("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image,shape,logger)
            result = self.model.predict(image)
            print("处理的结果是:",result)
            return result


class SMModel(PublicModelInterface):

    def __init__(self, model_path, model_inputs=None):
        self.model = keras.models.load_model(model_path)
        self.info = self.model.to_json()
    def get_info(self):
        return self.info

    def predict(self, data):
        shape = savemodel_input_shape(self.info,logger)
        print("该模型的输入形状是:", shape)
        if data["type"] == "b64":
            # 如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            print("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)
            img_b64 = universal_image_process(img_b64, shape, logger)
            result = self.model.predict(img_b64)
            # print("预测结果是:",result)
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content
                print("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image, shape, logger)
            np.set_printoptions(suppress=True)#savedmodel返回的是科学计数法，取消科学计数法
            result = self.model.predict(image)
            print("处理的结果是:", result)
            return result

class OnnxModel(PublicModelInterface):

    def __init__(self, model_path, model_inputs=None):
        self.session = onnxruntime.InferenceSession(model_path)

    def predict(self, data):
        shape = self.session.get_inputs()[0].shape[1:]
        if data["type"] == "b64":
            #如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            print("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)
            img_b64 = universal_image_process(img_b64,shape,logger)
            input = self.session.get_inputs()[0].name
            output = self.session.get_outputs()[0].name
            result = self.session.run([output], {input: img_b64})[0]
            # print("预测结果是:",result)
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content
                print("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image,shape,logger)
            input = self.session.get_inputs()[0].name
            output = self.session.get_outputs()[0].name
            result = self.session.run([output], {input: image})[0]
            print("处理的结果是:",result)
            return result
    def get_info(self):
        #输出模型的输入层、输入shape、输出层、输出shape
        input_layer = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        output_layer = self.session.get_outputs()[0].name
        output_shape = self.session.get_outputs()[0].shape
        info = {"input_layer":input_layer,"input_shape":input_shape,"output_layer":output_layer,
                "output_shape":output_shape}
        info = json.dumps(info)
        return info



#运行pb和ckpt模型需要额外的参数，主要包括tensor或op的name，以及feedict的key值
class CkptModel(PublicModelInterface):
    def __init__(self,model_path,model_inputs=None):
        pass

    def predict(self, data):
        pass

    def get_info(self):
        pass

class PbModel(PublicModelInterface):

    def __init__(self,model_path,model_inputs=None):
        pass

    def predict(self, data):
        pass

    def get_info(self):
        pass

class PthModel(PublicModelInterface):
    def __init__(self,model_path,model_inputs=None):
        pass

    def predict(self, data):
        pass

    def get_info(self):
        pass

