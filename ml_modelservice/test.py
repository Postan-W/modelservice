import os
import sys
import traceback
import logging
from base import spark
from load_and_predict_docker import Predict
from exception_enum import ExceptionEnum
import download_model
import json
import snappy
from pyspark.ml import PipelineModel
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.classification import JavaClassificationModel
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaParams, JavaWrapper
from zipfile import ZipFile
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
from load_and_predict_docker import Predict
local_model_path = "./chinamobile/model/model"
DISPLAY = "displayName"
NAME = "name"
model = None
pipeline_model = None
from flask import Flask, jsonify, request
def func():
    try:
        global model
        print("尝试加载PipelineModel")
        model = PipelineModel.load(local_model_path)  # 加载模型
        print("加载pipeline模型成功")
    except:
        try:
            # H2O模型必须走这里
            from pysparkling.ml import H2OMOJOSettings, H2OMOJOModel
            print("从加载PipelineModel的try中跳出")
            print("在except的try中尝试加载H2OMOJOModel")
            settings = H2OMOJOSettings(withDetailedPredictionCol=True)
            model = H2OMOJOModel.createFromMojo(local_model_path + '/mojo_model', settings)
        except:
            global pipeline_model
            print("从加载H2OMOJOModel的try中跳出")
            print("尝试加载XGBModel")
            # model = XGBoostClassificationModel.load(local_model_path)
            model = load_xgb_model(local_model_path, m_type='XGBoostClassificationModel')
            if not model:
                logging.error('XGBoostClassificationModel没有加载成功')
            pipeline_model = load_xgb_model(local_model_path, "PipelineModel")
            if not pipeline_model:
                logging.error('XGB需要的pipelinemodel没有加载成功')
                logging.error(pipeline_model)

    return model,pipeline_model


model,pipeline_model = func()

from define_spark import df

def XGB_predic_test():
    df.show()
    predictions = pipeline_model.transform(df)
    result = model.transform(predictions)
    print("预测的结果为:",result)

def chinamobile_predict():
    df.show()
    result = model.transform(df)
    print(result)

chinamobile_predict()