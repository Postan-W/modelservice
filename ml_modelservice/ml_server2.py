import os
import sys
import traceback
import logging
from load_and_predict_docker import Predict
from exception_enum import ExceptionEnum
import download_model
import json
from pyspark.ml import PipelineModel
from XGBoost_classification import XGBoostClassificationModel
import ast
import tornado
import tornado.escape
import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.log
import tornado.options
import tornado.web
from tornado.web import url, RequestHandler
from XGBoost_model_load import  load_xgb_model
from base import spark
import logging
import os
import sys
import requests
from flask import Flask, jsonify, request
import json
import traceback

DISPLAY = "displayName"
NAME = "name"

#创建服务
app = Flask(__name__)

class ChinaMobileModel():
    def __init__(self):
        local_model_path = "model/model"
        model = PipelineModel.load(local_model_path)  # 加载模型
        print("pipelinemodel加载完毕")
        self.model = model

    #预测输入数据的类型是一个列表，列表的每一个元素就是一个dict类型的要预测的数据
    def predict(self,data:list)->list:
        try:
            result = Predict.predict_by_model(data, model=self.model)
        except Exception as e:
            print("模型预测出错:",e)
        return result

model = ChinaMobileModel()
# model.predict([{'c_ZERO': 0, 'c_MEAN': 0}])


api_addr = "/capabilityname/v1/request"
@app.route('/' + api_addr, methods=['POST'])
def route_predict():
    try:
        data = json.loads(request.data)#POST的参数请用data接收。GET的请求参数请用args接收。
        return str()
    except Exception as e:
        traceback.print_exc()
        app.logger.error(e)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
