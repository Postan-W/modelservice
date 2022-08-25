#coding=utf-8
"""
@Author : wmingzhu
@Annotation : Tensorflow2.0，cpu版本。采用这个版本是因为：1.推理的对象是用户上传的一张图片，计算量小不需要GPU 2.GPU版本的Tensor flow跟镜像所在宿主机的GPU驱动版本以及CUDA版本等有关联，镜像移植能力差 3.tf.compat.v1可以处理v1版本的Tensor flow的模型加载与预测的事务，这样相当于将两个类型集成到一个镜像中，比分开部署简单
"""

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
from local_utilities import h5_resize_bs64,h5_input_shape,universal_image_process,savemodel_input_shape,find_latest
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

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
}

#抽象基类，限定了模型类要实现预测、返回模型信息这两个方法。目前第二个函数不需要实现
class PublicModelInterface(abc.ABC):

    @abc.abstractmethod
    def predict(self, data):
        pass

    @abc.abstractmethod
    def get_info(self):
        pass


class H5Model(PublicModelInterface):

    def __init__(self, modelpath,model_inputs=None):

        self.model = tf.keras.models.load_model(modelpath)
        #这是由用户从前端传来的模型输入
        self.shape = model_inputs
        print("用户输入的shape为", self.shape)
        try:
            self.info = self.model.to_json()
            logging.info("to_json()可以调用来初始化模型对象")
            #目前发现低版本的keras保存的h5模型，使用to_json()时会报错
        except:
            logging.info("to_json()方法无法调用，放弃使用其来初始化对象")
            self.info = "H5Model"
        logging.info("模型基本信息:::")
        logging.info(self.info)

    def get_info(self):
        return self.info

    def predict(self, data):
        try:
            shape = h5_input_shape(self.model.to_json(), logger)
            logging.info("to_json()方法可以调用，来解释模型输入shape")
        except:
            logging.info("to_json()方法无法调用，只能使用用户输入的shape数据")
            shape = self.shape
        print("该模型的输入形状是:",shape)
        if data["type"] == "b64":
            #如果b64以形如data:image/jpg;base64,开头，一定要把“data:image/jpg;base64,“这个前缀去掉，不然出错
            logging.info("处理的是b64图片")
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)
            img_b64 = universal_image_process(img_b64,shape,logger)
            result = self.model.predict(img_b64)
            # print("预测结果是:",result)
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content#直接得到bytes图片
                logging.info("获取网络图片成功")
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
            logging.info("处理的是b64图片")
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
                logging.info("获取网络图片成功")
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
        logging.info("初始化的是tf2或者keras保存的SavedModel模型")
        self.shape = model_inputs
        print("用户输入的shape为",self.shape)
        try:
            self.info = self.model.to_json()
            logging.info("to_json()方法可以使用，来初始化模型对象")
        except:
            logging.info("to_json()方法无法调用，放弃使用其初始化模型对象")
            self.info = "SMModel"
    def get_info(self):
        return self.info

    def predict(self, data):
        try:
            shape = savemodel_input_shape(self.model.to_json(), logger)
            print("解析得到的shape是:",shape)
            logging.info("to_json()方法可以调用，来解析模型输入shape")
        except:
            logging.info("to_json()方法无法调用，只能使用用户输入的shape数据")
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

#tensorflow静态图生成的SavedModel模型需要知道signature签名的关键键值来获取所需张量
class SavedModelTf1(PublicModelInterface):
    #将Session对象定义为类变量
    sess = tf.compat.v1.Session()

    def __init__(self,model_path,model_inputs=None):
        self.model = tf.compat.v1.saved_model.loader.load(SavedModelTf1.sess, [tf.compat.v1.saved_model.tag_constants.SERVING], model_path)#tag值这里默认为SERVING，有不妥
        self.shape = model_inputs
        logging.info("这是tf1版本的SavedModel模型，需要知道签名信息来还原图")

    def predict(self, data=None):
        """
        对下面try...except的解释:
        try的部分写的是通过前端传来的输入输出张量名称来获取张量并进行预测。
        except部分写的是如果没有获得这个信息，那么就按照图灵引擎平台流上产生的模型所具有的固定signature来解析。
        """
        #为了处理平台上产生的YOLO，加一层异常处理专门处理YOLO
        is_yolo = False
        try:
            signature = self.model.signature_def['prediction_signature']
            logging.info("yolo模型的图定义信息加载完成")
            input_tensor_name = signature.inputs['input_img'].name
            boxes_tensor_name = signature.outputs['boxes'].name
            input_image_shape_name = signature.inputs['input_image_shape'].name
            scores = signature.outputs['scores'].name  # scores就是置信度
            classes = signature.outputs['classes'].name
            print(input_tensor_name, boxes_tensor_name, input_image_shape_name, scores, classes)

            input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(input_tensor_name)
            boxes_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(boxes_tensor_name)
            input_image_shape_tesor = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_shape_name)
            scores_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(scores)
            classes_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(classes)
            logging.info("yolo模型还原成功")
            is_yolo = True

        except:
            try:
                input_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(data["inputTensor"])
                output_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name(data["outputTensor"])
                logging.info("传来的张量名称正确，已成功生成输入输出张量")
            except:
                logging.info("没有获得输入输出张量信息，只能按照固定格式解析")
                signature_def = self.model.signature_def
                signature_def = signature_def["test_signature"]
                input_tensor_name = signature_def.inputs['input_x'].name
                output_tensor_name = signature_def.outputs['outputs'].name
                print(input_tensor_name, output_tensor_name)
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
            img_b64 = universal_image_process(img_b64,shape,logger)#最后得到的数据是在shape基础上，前面加个样本维，并且样本维的值为1
            np.set_printoptions(suppress=True)# savedmodel返回的是科学计数法形式，取消科学计数法
            if is_yolo:
                logging.info("yolo的结果")
                result = SavedModelTf1.sess.run([boxes_tensor,scores_tensor,classes_tensor],feed_dict={input_tensor:img_b64,input_image_shape_tesor:[shape[0],shape[1]]})
            else:

                result = SavedModelTf1.sess.run(output_tensor, feed_dict={input_tensor:img_b64})
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
            if is_yolo:
                logging.info("yolo的结果")
                result = SavedModelTf1.sess.run([boxes_tensor, scores_tensor, classes_tensor],
                                                feed_dict={input_tensor: image,
                                                           input_image_shape_tesor: [shape[0], shape[1]]})
            else:
                result = SavedModelTf1.sess.run(output_tensor, feed_dict={input_tensor:image})
            print("处理的结果是:", result)
            return result

    def get_info(self):
        pass

#默认使用global_step最大的文件
class CkptModel(PublicModelInterface):
    sess = tf.compat.v1.Session()

    def __init__(self,model_path,model_inputs=None):
        meta_file = find_latest(model_path)
        latest_ckpt = os.path.splitext(meta_file)[0]
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(CkptModel.sess,latest_ckpt)
        self.graph = tf.compat.v1.get_default_graph()
        logging.info("成功还原当前图")
    def predict(self, data):
        input_tensor = self.graph.get_tensor_by_name(data["inputTensor"])
        output_tensor = self.graph.get_tensor_by_name(data["outputTensor"])
        logging.info("成功还原出输入输出张量")
        # 获取输入张量的形状
        input_shape = tuple(input_tensor.shape)
        # 因为shape总是类似（None,32,32,1)这种形式，需要转为类似[32,32,1]这种形式
        print(tuple(input_shape), type(tuple(input_shape)))
        shape = []
        for element in input_shape:
            if element != None:
                shape.append(element)
        logging.info(f'输入的形状是:{shape}')
        if data["type"] == "b64":
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)  # 得到bytes形式的图片
            img_b64 = universal_image_process(img_b64, shape, logger)  # 最后得到的数据是在shape基础上，前面加个样本维，并且样本维的值为1
            result = CkptModel.sess.run(output_tensor, feed_dict={input_tensor:img_b64})
            logging.info(f'预测的结果是:{result}')
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content#返回的是bytes类型；text返回的是str类型
                logger.info("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image, shape, logger)
            result = CkptModel.sess.run(output_tensor, feed_dict={input_tensor:image})
            logging.info(f'预测的结果是:{result}')
            return result
    def get_info(self):
        pass


class PbModel(PublicModelInterface):
    sess = tf.compat.v1.Session()
    def __init__(self,model_path,model_inputs=None):
        f = tf.compat.v1.gfile.FastGFile(model_path,'rb')
        self.graph_def = tf.compat.v1.GraphDef()
        self.graph_def.ParseFromString(f.read())#还原图

    def predict(self, data):
        input_tensor,output_tensor = tf.compat.v1.import_graph_def(self.graph_def,return_elements=[data["inputTensor"],data["outputTensor"]],name="")
        logging.info("成功还原出输入输出张量")
        # 获取输入张量的形状
        input_shape = tuple(input_tensor.shape)
        # 因为shape总是类似（None,32,32,1)这种形式，需要转为类似[32,32,1]这种形式
        print(tuple(input_shape), type(tuple(input_shape)))
        shape = []
        for element in input_shape:
            if element != None:
                shape.append(element)
        logging.info(f'输入的形状是:{shape}')
        if data["type"] == "b64":
            logger.info("处理的是b64图片")
            img_b64 = data["b64"]
            img_b64 = base64.b64decode(img_b64)  # 得到bytes形式的图片
            img_b64 = universal_image_process(img_b64, shape, logger)  # 最后得到的数据是在shape基础上，前面加个样本维，并且样本维的值为1
            result = PbModel.sess.run(output_tensor, feed_dict={input_tensor:img_b64})
            logging.info(f'预测的结果是:{result}')
            return result
        if data["type"] == "url":
            try:
                image = requests.get(data["url"]).content#返回的是bytes类型；text返回的是str类型
                logger.info("获取网络图片成功")
            except Exception as e:
                logger.info(e)
            logger.info("处理的是网络图片")
            image = universal_image_process(image, shape,logger)
            result = PbModel.sess.run(output_tensor,feed_dict={input_tensor:image})
            logging.info(f'预测的结果是:{result}')
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









