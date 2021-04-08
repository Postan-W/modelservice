#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: GolLong
# @Time  :2019/1/16 16:15
import base64
import io
import logging
import os
import json
import sys

import jsonpath
import numpy as np
import requests
import tornado
import tornado.escape
import tornado.httpclient
import tornado.httpserver
import tornado.ioloop
import tornado.log
import tornado.options
import tornado.web
from PIL import Image
from tornado.web import url, RequestHandler


# tornado.options.log_file_prefix = os.path.join(os.path.dirname(__file__), 'tornado.log')

get_headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"}

dims_list = None
def get_dims():
    # 解析模型输入格式
    metadata_url = 'http://localhost:8505/v1/models/' + os.environ.get('MODEL_NAME') + '/metadata'
    # metadata_url = 'http://10.129.16.3:8505/v1/models/lenet/metadata'
    logging.info('模型格式路径：%s', metadata_url)
    response = requests.get(metadata_url)
    json_obj = tornado.escape.json_decode(response.text)
    logging.info('获取模型格式：%s', str(json_obj))
    partten = '$.metadata.signature_def.signature_def.test_signature.inputs..dim[*].size'
    dims_list = jsonpath.jsonpath(json_obj, partten)
    logging.info('解析模型格式：%s', str(dims_list))
    # image_width, image_height, image_channle = int(dims_list[1]), int(dims_list[2]), int(dims_list[3])
    # return dims_list
    # # return image_width, image_height, image_channle
    #
    #
    # dims_list = get_dims()
    dims_list = [int(i) for i in dims_list]
    logging.info(dims_list)
    if dims_list.count(-1) == 1:
        dims_list = [i for i in dims_list if i != -1]
        return dims_list
    elif dims_list.count(-1) >= 2:
        start_index = dims_list.index(-1)
        end_index = dims_list.index(-1,start_index+1)
        dims_list = dims_list[start_index + 1:end_index]
        return dims_list
    else:
        logging.error('解析模型格式异常,该值必须大于0:'+str(dims_list.count(-1)))
        # logging.error("模型暂时没有启动，请稍后再试")
        return None

class ServingHandler(RequestHandler):


    def post(self):
        self.set_header("Content-Type", "application/json; charset=UTF-8")
        # 用户token、用于验证权限
        try:
            token = self.get_query_argument("token")
            logging.info('当前请求token：%s', token)

            jsonbyte = self.request.body
            jsonstr = jsonbyte.decode('utf8')
            jsonobj = json.loads(jsonstr)  # 将字符串转为json对象
            type = jsonobj.get('type')  # 就可以用api取值

            logging.info('当前请求type：%s', type)

        except Exception as e:
            logging.error('缺少参数')
            logging.error(e)
            self.write({'error': '缺少参数'})
            return

        try:
            token_result = self.check_token(token, '')

        except Exception as e:
            logging.error('token认证服务异常')
            logging.error(e)
            self.write({'error': 'token认证服务异常'})
            return
        logging.info('token认证结果：%s', token_result)

        if json.loads(token_result).get('code', None) != 1000:
            self.write({'error': 'token认证失败'})
            return

        # base64方式
        if type == 'b64':
            # 获取base64字符串
            try:
                body_b64 = jsonobj.get('b64')
                if not body_b64:
                    logging.error('缺少参数：b64')
                    self.write({'error': '缺少参数：b64'})
                    return
                # 获取文件上传
                base64_img = base64.urlsafe_b64decode(body_b64)
                image = io.BytesIO(base64_img)
                logging.info(image)
                img = Image.open(image)  # 无法解析的base64编码
                # try:
                #     image_width, image_height, image_channle = self.get_dims()
                # except Exception as e:
                #     logging.error('解析模型输入格式发生错误')
                #     logging.error(e)
                #     self.write({'error': '解析模型输入格式发生错误'})
                #     return
                # base_image = np.asarray(np.resize(img, (1, image_width, image_height, image_channle)))
                global dims_list
                if not dims_list:
                    dims_list = get_dims()
                base_image = np.asarray(np.resize(img, (1, *dims_list)))
                instances = base_image.tolist()
                predict_request = json.dumps(
                    {"signature_name": "test_signature", "instances": instances})  # 字符串格式没问题，但和python base64算法标准不一样
            except Exception as e:
                logging.error({'error': str(e)})
                self.write({'error': str(e)})
                return
        # url方式
        elif type == 'url':
            # 获取网页图片路径
            try:
                image_url = jsonobj.get('url')
                if not image_url:
                    logging.error('缺少参数：url')
                    self.write({'error': '缺少参数：url'})
                    return
                # 获取文件上传
                logging.info('接收的图片URL：%s', image_url)
                response = requests.get(image_url, headers=get_headers)
                logging.info(response)
                image_file = io.BytesIO(response.content)
                logging.info(image_file)
                img = Image.open(image_file)
                # try:
                #     image_width, image_height, image_channle = self.get_dims()
                # except Exception as e:
                #     logging.error(e)
                #     self.write({'error': str(e)})
                #     return
                #
                # url_image = np.asarray(np.resize(img, (1, image_width, image_height, image_channle)))

                global dims_list
                if not dims_list:
                    dims_list = get_dims()
                url_image = np.asarray(np.resize(img, (1, *dims_list)))
                instances = url_image.tolist()
                predict_request = json.dumps({"signature_name": "test_signature", "instances": instances})
            except Exception as e:
                logging.error({'error': str(e)})
                self.write({'error': str(e)})
                return

        # data数组方式
        elif type == 'data':
            try:
                data = jsonobj.get('data')
                if not data:
                    logging.error('缺少参数：data')
                    self.write({'error': '缺少参数：data'})
                    return
                data = json.loads(data)
                logging.info('接收的数组字符串数据：%s', str(data))
                predict_request = json.dumps({"signature_name": "test_signature", "instances": data})
            except Exception as e:
                logging.error(e)
                self.write({'error': str(e)})
                return

        else:
            self.write({'error': '非法的参数：type'})
            return

        # 构造模型路径
        server_url = 'http://localhost:8505/v1/models/' + os.environ.get('MODEL_NAME') + ':predict'
        # server_url = 'http://10.129.16.3:8505/v1/models/lenet:predict'
        logging.info(server_url)
        response = requests.post(server_url, data=predict_request)
        logging.info('模型预测结果：%s', response.text)
        self.write(response.text)
        #res_dict = json.loads(response.text)
        # 应该只有一个key，predictions
        #key = list(res_dict.keys())[0]
        # 应该是二维数组，且最外层只有一个元素，取内层数组中最大值的索引
        #value = res_dict.get(key)[0]
        #idx = value.index(max(value))
        #return self.write({key: "预测的序号为{}，请参照数据集明确具体类别".format(idx)})

    def check_token(self, token, request_param):

        url = os.environ.get('XQUERY_ADDR') + '/dsModel/serviceApply/tokenVerify'
        logging.info('开始校验token，地址为：%s', url)
        body = {
            'callToken': token,
            'serviceId': os.environ.get('MODEL_SERVICE_ID'),
            'requestParam': str(request_param)
        }
        headers = {'content-type': 'application/json'}
        logging.info('参数为：%s', body)
        response = requests.post(url, data=json.dumps(body), headers=headers)
        return response.text


class LogFormatter(tornado.log.LogFormatter):
    def __init__(self):
        super(LogFormatter, self).__init__(
            fmt='%(color)s[%(asctime)s %(filename)s:%(funcName)s:%(lineno)d %(levelname)s]%(end_color)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )


if __name__ == "__main__":
    tornado.options.parse_command_line()
    [i.setFormatter(LogFormatter()) for i in logging.getLogger().handlers]



    logging.info('MODEL_NAME:%s', os.environ.get('MODEL_NAME'))
    logging.info('XQUERY_ADDR:%s', os.environ.get('XQUERY_ADDR'))
    logging.info('MODEL_SERVICE_ID %s:', os.environ.get('MODEL_SERVICE_ID'))
    logging.info('API_ADDR: %s', os.environ.get('API_ADDR'))

    if not (os.environ.get('MODEL_NAME') and os.environ.get('XQUERY_ADDR') and os.environ.get(
            'MODEL_SERVICE_ID') and os.environ.get('API_ADDR')):
        logging.error('环境变量不存在')
        sys.exit(1)

    app = tornado.web.Application(
        [
            url(r"/" + os.environ.get('API_ADDR'), ServingHandler),
            # url(r"/lenet", ServingHandler),
        ],
        debug=False
    )

    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8501)
    tornado.ioloop.IOLoop.current().start()
