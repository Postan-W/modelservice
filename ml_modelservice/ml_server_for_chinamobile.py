# -*- coding: UTF-8 -*-
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
from define_spark import spark
import json
import base64
aa = os.getcwd()
# 设置加载路径
sys.path.append(aa)
sys.path.append(aa + '/machine_learning')


config_ = {
    'download_model_zip_path': '/model.zip',
    'local_model_path': 'model',
    'transform_json_path': 'columns/datatype/'
}

# 下载模型zip路径，默认为根目录下model.zip文件。
download_model_zip_path = config_.get('download_model_zip_path')
# 解压模型路径，默认为根目录，与压缩包model.zip同级
unzip_path = os.path.split(download_model_zip_path)[0]
# local_model_path = os.path.join(unzip_path, config_['local_model_path'])
local_model_path = "/root/machine_learning/chinamobile/model/model"
# 模型下，用来转换字段的json
transform_json_path = config_['transform_json_path']
final_transform_json_path = None
DISPLAY = "displayName"
NAME = "name"
model = None
pipeline_model = None
model_json = None


logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


class Serving_Handler(RequestHandler):

    async def post(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        print("进入请求处理函数")
        request_body = self.get_request_body()
        b64_para = request_body["X-Server-Param"]
        print("原本的b64参数是:",b64_para)
        decode_para = base64.b64decode(b64_para)
        decode_para = json.loads(decode_para)#key值需要是双引号，否则loads字典报错
        print("解码后的参数为:",decode_para)
        print("参数类型为:",type(decode_para))
        predict_data = decode_para["prediction_data"]
        print("amount的值是：",predict_data["amount"])
        logging.info(f'用户输入的数据的类型为 {type(predict_data)}')
        logging.info(f'用户输入的数据为 {predict_data}')
        # 由于模型使用的字段，可能与用户输入不一致，故需要转换
        converted_src = self.convert_field(predict_data, DISPLAY, NAME)
        if not converted_src:
            return
        try:
            # 预测
            result = Predict.predict_by_model(json_data=converted_src, model=model)
        except:
            try:
                data = spark.createDataFrame((converted_src))
                logging.info("下面是XGB接收的data：")
                logging.info(data)
                columns = data.columns
                # 若传入的unlabeled_data包含prediction列，则删除
                if 'prediction' in columns:
                    columns = columns[:-1]
                    print(columns)
                    data = data.select(*columns)
                print('unlabeled data display---------------------------------------------')
                data.show()
                predictions = pipeline_model.transform(data)
                result = model.transform(predictions)
                #这个result是pyspark的DataFrame类型，这里把它转化为dict
                result = list(map(lambda row: row.asDict(), result.collect()))
                print("预测结果为：",result)
            except:
                logging.error(f'预测失败 ===> {traceback.format_exc()}')
                ExceptionEnum.PREDICT_FAILED.value.update({"X-Code": 1010104,"X-Desc":"invalid body"})
                self.write(ExceptionEnum.PREDICT_FAILED.value)
                return

        logging.info(f'预测成功，结果为 {result}')

        # 将字段再转换回去
        result_data = self.convert_field(result, NAME, DISPLAY)
        print("经过转换的预测结果是：",result_data)
        print("经过转换的预测结果的类型是：",type(result_data))
        #因为features、probablity、rawPrediction这几个key的值无法被序列化会导致错误，这里去掉它们
        try:
            for element in result_data:
                del element['FEATURES']
                del element['RAWPREDICTION']
                del element['PROBABILITY']
        except:
            print("该模型不需要删除FEATURES等键值")
        print("结果是：",result_data)
        if not result_data:
            return
        x_server_prarm = str({"sid":decode_para["csid"],"result":result_data}).encode('utf-8')
        x_server_prarm = base64.b64encode(x_server_prarm)
        ExceptionEnum.SUCCESS.value.update({"X-Server-Param": str(x_server_prarm),"X-Code":1000000,"X-Desc":"success","X-IsSync":True})
        self.write(ExceptionEnum.SUCCESS.value)
        return

    def get_request_body(self):
        request_body = self.request.body.decode('utf8')#request.body直接得到Python字典
        # 将字符串转为python对象
        try:
            predict_data = ast.literal_eval(request_body)
        except:
            # 如果转化不成功，就取原值
            predict_data = request_body
        logging.info(f'预测数据 ：{predict_data}')
        return predict_data

    def convert_field(self, data, src_key, dst_key):
        """转换字段"""
        try:
            converted_result = convert_field(data, src_key, dst_key)
        except Exception as e:
            logging.error(f'字段转换失败 ===> {traceback.format_exc()}')
            ExceptionEnum.FIELD_CONVERT_FAILED.value.update({"data": e.args[0]})
            self.write(ExceptionEnum.FIELD_CONVERT_FAILED.value)
            return
        return converted_result

    @staticmethod
    async def check_token(token, request_param):
        """
        校验token
        :param token: 要校验的token
        :param request_param: 调用参数
        :return:
        """
        check_token_url = os.environ.get('XQUERY_ADDR') + '/dsModel/serviceApply/tokenVerify'
        logging.info(f'开始校验token，地址为 ===> {check_token_url}')
        body = {
            "callToken": token,
            "serviceId": os.environ.get('MODEL_SERVICE_ID'),
            "requestParam": str(request_param)
        }
        headers = {'content-type': "application/json"}
        logging.info(f'校验token的参数为 ===> {body}')
        # tornado框架的异步http客户端
        http = tornado.httpclient.AsyncHTTPClient()
        response = await http.fetch(check_token_url, method='POST', body=json.dumps(body), headers=headers)
        logging.info(f'token_result ===> {response.body}')
        return json.loads(response.body).get('code') == 1000


def upper_dict_key(src_dict):
    """将字典的key进行大写"""
    res_dict = dict()
    for k, v in src_dict.items():
        res_dict[k.upper()] = v
    return res_dict


def convert_dict(model_fields, value_dict, src_key, dst_key):
    """
    提供字典转化功能
    :param model_fields: [
                            {
                                "ord":"0",
                                "typeClassName":"java.lang.String",
                                "code":"column_0",
                                "originalType":"12",
                                "sparkType":"string",
                                "displayName":"ZERO列",
                                "name":"column_0",
                                "originalTypeName":"BigInteger"
                            }
                        ]
                        结构为：列表下嵌套字典，以下称子字典为 `field`
    :param value_dict: 带有数值的字典，例如：{'zero列': 0}
    :param src_key: 在 field 中，需要被转化的key
    :param dst_key: 在 field 中，转化的目标key
    转化过程为，遍历 model_fields，若 field 中的 src_key 所对应的值存在于 value_dict 的键中，那么即匹配成功，
    取 field 的 dst_key 所对应的值为键，value_dict 中的值（匹配所使用的的值）为值（自动删除源值），放入 value_dict 中
    eg: {'zero列': 0}，==> {'column_0': 0}
    """
    value_dict = upper_dict_key(value_dict)
    for field in model_fields:
        # 获取src_key 在field中对应的值，例如：src_key 为 displayName，field_src_value = ZERO列
        field_src_value = field.get(src_key).upper()
        # 判断 field_src_value 是否存在于 value_dict中
        if field_src_value in value_dict.keys():
            # 若存在，将value_dict中键为 field_src_value的值弹出，以上例，value_dict 将弹出 ZERO列，
            # 设dst_key 为name，那么value_dict将新增一个键为column_0，值为value_dict弹出的ZERO列对应的值
            value_dict[field.get(dst_key)] = value_dict.pop(field_src_value)
    return value_dict


def convert_field(src_data, src_key, dst_key):
    """
    将展示的名称(displayName)转换成模型需要的名称(name)
    模型将字段对应关系存入json文件，目前路径为 columns/datatype/*.json
    json文件内容示例：
    {
        "dataSetId":"columns",
        "fieldDefs":[
                        {
                            "ord":"0",
                            "typeClassName":"java.lang.String",
                            "code":"column_0",
                            "originalType":"12",
                            "sparkType":"string",
                            "displayName":"ZERO列",
                            "name":"column_0",
                            "originalTypeName":"BigInteger"
                        }
                    ]
    }
    fieldDefs结构为：列表下嵌套字典，以下称子字典为 `field`
    :param src_data: 传入的参数，现支持：字典，列表，元祖
    :param src_key: 在 field 中，需要被转化的key
    :param dst_key: 在 field 中，转化的目标key
    例如：输入数据为 {'zero列': 0} ==> {'column_0': 0}
    :return:
    """
    logging.info(f"开始转换字段数据 {src_key} to {dst_key}")
    converted_fields = list()
    model_fields = model_json.get('fieldDefs')
    if isinstance(src_data, (list, tuple)):
        print("进入到了列表的形式中")
        [converted_fields.append(convert_dict(model_fields, data, src_key, dst_key))
         for data in src_data]
    elif isinstance(src_data, dict):
        print("进入到了字典的形式中")
        converted_fields.append(convert_dict(model_fields, src_data, src_key, dst_key))
    else:
        raise TypeError(f"暂不支持的参数类型：{type(src_data)}")
    logging.info(f"字段转换成功({src_key} to {dst_key}) ===> {converted_fields}")
    return converted_fields


def get_model_path(p):
    """
    找到真实下载后的模型路径
    :param p:
    :return:
    """
    listdir = os.listdir(p)
    for child_dir in listdir:
        child_dir_full = os.path.join(p, child_dir)
        if os.path.isdir(child_dir_full):
            if child_dir == 'metadata' and p.endswith("model"):
                global local_model_path
                local_model_path = p
                break
            else:
                get_model_path(child_dir_full)


def get_jsonfile_fullname():
    """
    获取目录下json文件，因为json文件名不规范，所以要获取
    :return:
    """
    model_path = os.path.split(local_model_path)[0]
    path = os.path.join(model_path, transform_json_path)
    dirs = os.listdir(path)  # 获取指定路径下的文件
    for i in dirs:  # 循环读取路径下的文件并筛选输出
        if os.path.splitext(i)[1] == ".json":  # 筛选json文件
            return os.path.join(path, i)


def init():
    # 加载模型
    try:
        global model
        print("尝试加载PipelineModel")
        model = PipelineModel.load(local_model_path)#加载模型
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
            model = load_xgb_model(local_model_path,m_type='XGBoostClassificationModel')
            if not model:
                logging.error('XGBoostClassificationModel没有加载成功')
            pipeline_model = load_xgb_model(local_model_path, "PipelineModel")
            if not pipeline_model:
                logging.error('XGB需要的pipelinemodel没有加载成功')
                logging.error(pipeline_model)

    global final_transform_json_path
    final_transform_json_path = get_jsonfile_fullname()

    # 读取json，model_json: 模型中存储的json
    with open(final_transform_json_path, encoding='utf-8') as f:
        global model_json
        model_json = json.load(f)


if __name__ == '__main__':
    # 初始化工作
    init()
    api_addr = "v1/request"
    app = tornado.web.Application(
        [
            url(api_addr if api_addr.startswith("/") else "/" + api_addr, Serving_Handler),
        ],
        debug=False
    )
    http_server = tornado.httpserver.HTTPServer(app)  #创建httpserver实
    http_server.listen(5000) # 监听端口
    logging.info("服务启动成功！")
    tornado.ioloop.IOLoop.current().start()  # 开启epoll
