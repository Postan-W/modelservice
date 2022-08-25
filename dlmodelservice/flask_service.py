#@Author : wmingzhu
#@Email : wangmingzhu@bonc.com.cn
#coding=utf-8
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from fnmatch import fnmatch
import requests
from flask import Flask, jsonify, request
import json
import traceback
import numpy as np
import re
from models import SMModelTf2, OnnxModel, H5Model, CkptModel,PbModel,PthModel,SavedModelTf1

app = Flask(__name__)
#---------------日志部分----------------------------------------
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler("./logs/flask_service_log.txt", maxBytes=1024*1024*20, backupCount=50, encoding="utf-8")
formatter = logging.Formatter(
        '%(asctime)s--文件名:%(filename)s--文件路径:%(pathname)s--函数名:%(funcName)s--行号:%(lineno)s--进程id:%(process)s--日志级别:%(levelname)s--日志内容:%(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logging.getLogger().addHandler(handler)
logging.getLogger().addHandler(console)
#-----------------------------------------------------------------------------------------

# 获取环境变量
xquery_addr = os.environ.get('XQUERY_ADDR')
model_service_id = os.environ.get('MODEL_SERVICE_ID')
api_addr = os.environ.get('API_ADDR')

if not (xquery_addr and model_service_id and api_addr):
    app.logger.error('缺少环境变量')
    sys.exit(1)

# model_inputs = os.environ.get('MODEL_INPUTS')
# #上面的model_inputs是一个环境变量的值，也就是字符串，我们需要列表形式的或者numpy数组形式的模型输入
# pattern = re.compile("\d+?(?=,|\])")
# model_inputs = pattern.findall(model_inputs)
# model_inputs = [int(element) for element in model_inputs]


# 验证token
def check_token(token, request_param):
    url = xquery_addr + '/dsModel/serviceApply/tokenVerify'
    app.logger.info('开始校验token，地址为：%s', url)
    body = {
        'callToken': token,
        'serviceId': model_service_id,
        'requestParam': str(request_param)
    }
    headers = {'content-type': 'application/json'}
    app.logger.info('参数为：%s', body)
    response = requests.post(url, data=json.dumps(body), headers=headers)
    return response.text

model_name = os.listdir("/models/")
entire_path = os.path.join("/models/",model_name[0])
model_path = entire_path
model_type = os.path.splitext(entire_path)[1]


if model_type == ".onnx":
    model = OnnxModel(model_path)
    app.logger.info("模型加载完毕->" + model_type[1:] + "模型")
elif model_type == ".h5":
    model = H5Model(model_path)
    app.logger.info("模型加载完毕->"+model_type[1:]+"模型")
elif model_type == ".ckpt":
    model = CkptModel(model_path)
elif model_type == ".pb":
    model = PbModel(model_path)
    app.logger.info("模型加载完毕->" + model_type[1:] + "模型")
elif model_type == ".pth" or model_type == ".pt":
    model = PthModel(model_path)
    app.logger.info("模型加载完毕->" + model_type[1:] + "模型")
elif model_name[0] == "model":
    #"/models/model/model"下面可能是ckpt文件也可能是SavedModel文件
    file_list = os.listdir("/models/model/model")
    is_SavedModel_tag = False
    for file in file_list:
        if file.endswith(".pb"):
            is_SavedModel_tag = True
            break
    if is_SavedModel_tag:
        try:
            model = SMModelTf2("/models/model/model")
            #如果是tf2或者keras保存的SavedModel模型，那么加载后，可以直接调用to_json函数
            model.model.to_json()
            app.logger.info("tf2版本的SavedModel模型加载完毕")
        except:
            print("上面调用to_json()函数失败，说明该模型是tf1格式的")
            model = SavedModelTf1("/models/model/model")
            app.logger.info("tf1版本的SavedModel模型加载完毕")
    else:
        app.logger.info("该模型为ckpt模型")
        model = CkptModel("/models/model/model")
        app.logger.info("ckpt模型加载完成")
else:
    app.logger.error('不支持当前模型格式:' + model_type)
    sys.exit(1)



@app.route('/' + api_addr, methods=['POST'])
def route_predict():
    try:
        data = json.loads(request.data)#POST的参数请用data接收得到Python字典格式数据。GET的请求参数请用args接收。
        token = data["token"]
        app.logger.info(token)
        token_result = check_token(token, "")
        if json.loads(token_result).get('code', None) != 1000:
            return jsonify({'error': 'token认证失败'})
        return jsonify({"result":str(model.predict(data))})
    except Exception as e:
        traceback.print_exc()
        app.logger.error(e)
        return jsonify({"error": str(e)})


@app.route('/' + api_addr + '/metadata', methods=['GET'])
def route_metadata():
    try:
        return model.get_info()
    except Exception as e:
        traceback.print_exc()
        app.logger.error(e)
        return jsonify({"error": str(e)})

#这个用来容器的HEALTHCHECK(通过docker ps查看status),没有其他作用,可以删除
@app.route('/successful')
def sucessfully():
    return "sucessfully!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
