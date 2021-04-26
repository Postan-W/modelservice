# -*- coding: utf-8  -*-
# @Author   :
# @time     : 2019/11/25 19:10:27
# File      : XGBoost_model_load.py
# Software  : PyCharm

import os
from pyspark.ml.pipeline import PipelineModel
from XGBoost_classification import XGBoostClassificationModel
from pyspark.ml.pipeline import DefaultParamsReader, PipelineSharedReadWrite
from base import spark

# 构建sparkContext，saprk是XGBoost_classification中导入jar的
sc = spark.sparkContext

"""
参数一：path，待加载的模型路径
参数二;m_type，模型类型：PipelineModel、XGBoostClassificationModel。其中，PipelineModel为带预测数据进行转换，包括生成
features向量列，然后在使用XGBoostClassificationModel对带预测集进行预测

返回值：
      model：model
      
使用示例：
>>>model_path = 'C:\\Users\\YUDAN\\Desktop\\xgb\\model'
>>>from XGBoost_model_load import load_xgb_model
>>>pipeline_model = load_xgb_model(model_path, "PipelineModel")
>>>pipeline_model
PipelineModel_ecbb40d2837f
>>>xgb_model = load_xgb_model(model_path, "")
XGBoostClassificationModel_ba649c5f7f2b

"""


def load_xgb_model(path, m_type):
    """

    :param path: model输入路径
    :param m_type: model类型
    :return: 输出对应model
    """
    # 获取model目录下metadata dict
    metadata = DefaultParamsReader.loadMetadata(path, sc)
    # stages_dir = os.path.join(path, "stages")  # stage路径
    stages_dir = path + "/stages"
    stage_uids = metadata['paramMap']['stageUids']  # metadata中model的uid、路径名
    stage_paths = {}  # 构建空dict
    # 循环遍历
    for index, stage_uid in enumerate(stage_uids):
        # 遍历model，获取相应的stages目录下model的路径
        stage_path = \
            PipelineSharedReadWrite.getStagePath(stage_uid, index, len(stage_uids), stages_dir)
        # stage_paths.append(stage_path)
        # 获取model，以及将相应路径写入字典
        key = stage_uid.split('_')[0]
        stage_paths[key] = str(stage_path)

    # stage_paths = load_xgb_model(path, sc)
    # 根据model type选择相应的model load方法，并返回相应的model
    # model type为PipelineModel、XGBoostClassificationModel
    if m_type == 'PipelineModel':
        model_path = stage_paths['PipelineModel']
        model = PipelineModel.load(model_path)
    else:
        model_path = stage_paths['xgbc']
        model = XGBoostClassificationModel.load(model_path)

    return model


if __name__ == '__main__':
    # model_path = 'C:\\Users\\YUDAN\\Desktop\\xgb\\model'
    model_path = 'hdfs://172.16.18.114:9000///AI/model/0b4f2c5a6ee14f2ba7f0a200fa4e6f4c/6f6b6ea4-47e2-4427-ab5f-a5728a3d6879/model_XGboost/version/1577102199722/model'
    xgb_model = load_xgb_model(model_path, m_type='XGBoostClassificationModel')