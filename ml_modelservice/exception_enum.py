# -*- coding: utf-8 -*-
from enum import Enum
_author_ = 'luwt'
_date_ = '2020/5/21 15:47'


class ExceptionEnum(Enum):
    SUCCESS = {'code': 1000, 'msg': 'ok'}
    NO_TOKEN = {'code': -1, 'msg': '缺少token'}
    CHECK_TOKEN_ERROR = {'code': -2, 'msg': 'token校验失败'}
    FIELD_CONVERT_FAILED = {'code': -3, 'msg': '字段转换失败'}
    PREDICT_FAILED = {'code': -4, 'msg': '预测失败'}


