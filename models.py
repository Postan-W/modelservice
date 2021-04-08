import abc
import base64
import io
import os
import onnxruntime
import requests
import torch
import tensorflow as tf
from pypmml import Model
import tensorflow as tf
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

        self.model = tf.keras.models.load_model(modelpath)
        #这是由用户从前端传来的模型输入
        self.shape = model_inputs
        print("用户输入的shape为", self.shape)
        try:
            self.info = self.model.to_json()
            print("to_json()可以调用来初始化模型对象")
            #目前发现低版本的keras保存的h5模型，使用to_json()时会报错
        except:
            print("to_json()方法无法调用，放弃使用其来初始化对象")
            self.info = "H5Model"
        logging.info("模型基本信息:::")
        logging.info(self.info)

    def get_info(self):
        return self.info

    def predict(self, data):
        try:
            shape = h5_input_shape(self.model.to_json(), logger)
            print("to_json()方法可以调用，来解释模型输入shape")
        except:
            print("to_json()方法无法调用，只能使用用户输入的shape数据")
            shape = self.shape
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


class PthModel(PublicModelInterface):
    def __init__(self, model_path, model_inputs=None):
        self.model = torch.load(model_path)
        self.info = self.model

    def predict(self, data):
        shape = [3,128,128]#暂时写死，输入shape无法解析，需要使用者传入
        print("该模型的输入形状是:", shape)
        if data["type"] == "b64":
            # 如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            print("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)
            img_b64 = universal_image_process(img_b64, shape, logger)
            result = self.model(img_b64)
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
            result = self.model(image)
            result = self.model(image)
            print("处理的结果是:", result)
            return result

    def get_info(self):
        return self.info

class SMModelTf2(PublicModelInterface):

    def __init__(self, model_path, model_inputs=None):
        self.model = tf.keras.models.load_model(model_path)
        print("初始化的是tf2或者keras保存的SavedModel模型")
        self.shape = model_inputs
        print("用户输入的shape为",self.shape)
        try:
            self.info = self.model.to_json()
            print("to_json()方法可以使用，来初始化模型对象")
        except:
            print("to_json()方法无法调用，放弃使用其初始化模型对象")
            self.info = "SMModel"
    def get_info(self):
        return self.info

    def predict(self, data):
        try:
            shape = savemodel_input_shape(self.model.to_json(), logger)
            print("解析得到的shape是:",shape)
            print("to_json()方法可以调用，来解析模型输入shape")
        except:
            print("to_json()方法无法调用，只能使用用户输入的shape数据")
            shape = self.shape
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

#下面定义图灵引擎平台流上产生savedModel(tf1形式)模型的加载类
class SavedModelFromFlow(PublicModelInterface):
    #将Session对象定义为类变量
    sess = tf.compat.v1.Session()

    def __init__(self,model_path,model_inputs=None):
        self.model = tf.compat.v1.saved_model.loader.load(SavedModelFromFlow.sess,[tf.compat.v1.saved_model.tag_constants.SERVING],model_path)
        self.shape = model_inputs
        print("这是来自图灵引擎平台流上的模型")
        print("流上的签名是写死的，所以这里就按照固定方式解析")

    def predict(self, data=None):
        signature_def = self.model.signature_def
        signature_def = signature_def["test_signature"]
        input_tensor_name = signature_def.inputs['input_x'].name
        output_tensor_name = signature_def.outputs['outputs'].name
        input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(input_tensor_name)
        output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(output_tensor_name)
        # 获取输入张量的形状
        input_shape = tuple(input_tensor.shape)
        #因为shape总是类似（None,32,32,1)这种形式，需要转为类似[32,32,1]这种形式
        print(tuple(input_shape),type(tuple(input_shape)))
        shape = []
        for element in input_shape:
            if element != None:
                shape.append(element)

        print("输入形状是：",shape)
        #下面开始处理
        if data["type"] == "b64":
            # 如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            print("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)#得到bytes形式的图片
            img_b64 = universal_image_process(img_b64,shape,logger)
            np.set_printoptions(suppress=True)  # savedmodel返回的是科学计数法形式，取消科学计数法
            result = SavedModelFromFlow.sess.run(output_tensor,feed_dict={input_tensor:img_b64})
            print("预测结果是:",result)
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content#返回的是bytes类型；text返回的是str类型
                print("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image, shape, logger)
            np.set_printoptions(suppress=True)#savedmodel返回的是科学计数法，取消科学计数法
            result = SavedModelFromFlow.sess.run(output_tensor,feed_dict={input_tensor:image})
            print("处理的结果是:", result)
            return result

    def get_info(self):
        pass

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



