import requests
import json
import base64
import math
import uuid
import time
import hashlib

# 下面的链接、端口等只是示例，请用真实值代替
url = "http://10.128.12.15"
interface_path = "/v1/request"
port = "5000"

# 完整链接
connection = url + ":" + port + interface_path


# 生成唯一id，用于得到csid
def getUUID():
    return "".join(str(uuid.uuid4()).split("-"))


# 生成请求头和csid
def create_header_scid(URL, APPID, APPKey):
    appid = APPID
    appKey = APPKey
    uuid = getUUID()
    # 24 + 32 + 8
    appName = URL.split('/')[-1]
    if len(appName) > 24:
        appName = appName[0:24]
    else:
        for i in range(24 - len(appName)):
            appName += "0"
    capabilityname = appName
    csid = appid + capabilityname + uuid
    tmp_xServerParam = {
        "appid": appid,
        "csid": csid
    }
    xCurTime = str(math.floor(time.time()))
    xServerParam = str(base64.b64encode(json.dumps(tmp_xServerParam).encode('utf-8')), encoding="utf8")
    # turn to bytes
    xCheckSum = hashlib.md5(bytes(appKey + xCurTime + xServerParam, encoding="utf8")).hexdigest()

    header = {
        "appKey": appKey,
        "X-Server-Param": xServerParam,
        "X-CurTime": xCurTime,
        "X-CheckSum": xCheckSum,
        "content-type": "application/json"
    }

    return header, csid


header, csid = create_header_scid(URL='http://117.132.181.235:9050/object_detection', APPID='', APPKey='')
"""
下面是请求的json格式,其值只是示例，调用者应使用真实数据：
csid：调用方的会话id，由调用方生成，须保证其唯一性
prediction_data：即高危号码识别模型预测所需要的数据，其值是一个json
"""
data = {"csid": csid,
        "prediction_data": {"amount": 0, "s3_flux": 0, "s2_flux": 0, "s1_flux": 0, "sc_1": 0, "sc_2": 5, "buka_cnt": 0,
                            "home_city": 591, "imei_ischang": 0, "model_desc_ischang": -5, "ic_no_num": 20201106101340,
                            "imei_num": 591, "call_called_num": 0, "rec_cnt": 1, "all_cnt": 202011060000000,
                            "brand_id": 0, "chan_cnt": 0, "create_time": 352225}}

# 请求的数据需要被转化成base64格式
data = base64.b64encode(json.dumps(data).encode('utf-8'))
data = data.decode('utf-8')

"""上面的请求数据data在下面又被包了一层json作为最终的请求数据，key也取的X-Server-Param，这个key名称和header里X-Server-Param一样了，实际上二者没有
任何关系，这是设计能力接口时的失误，不影响功能。
"""
request_body = {"X-Server-Param": data}


def send():
    try:
        # 响应数据
        response = requests.post(connection, data=json.dumps(request_body), headers=header).text
        response = json.loads(response)
        """
        能力接口返回的处理结果示例：
        {"code": 1000, "msg": "ok", "X-Server-Param": 
        "b'eydzaWQnOiAnMzc4ODk2NTUnLCAncmVzdWx0JzogW3snUFJPQkFCSUxJVFknOiB7J2FycmF5JzogWzAuMDEx
        NjE2MDkxMiwgMC45ODgzODM5MDg4XSwgJ3ZhbHVlcyc6IFswLjAxMTYxNjA5MTIsIDAuOTg4MzgzOTA4OF19LCAnUFJFRElDVElPT
        ic6IDEuMCwgJ2Ftb3VudCc6IDAsICdzM19mbHV4JzogMCwgJ3MyX2ZsdXgnOiAwLCAnczFfZmx1eCc6IDAsICdzY18xJzogMCwgJ3NjXzInOiA
        1LCAnYnVrYV9jbnQnOiAwLCAnaG9tZV9jaXR5JzogNTkxLCAnaW1laV9pc2NoYW5nJzogMCwgJ21vZGVsX2Rlc2NfaXNjaGFuZyc6IC01LCAnaWNfbm9f
        bnVtJzogMjAyMDExMDYxMDEzNDAsICdpbWVpX251bSc6IDU5MSwgJ2NhbGxfY2FsbGVkX251bSc6IDAsICdyZWNfY250JzogMSwgJ2FsbF9jbnQnOiAyMDIwM
        TEwNjAwMDAwMDAsICdicmFuZF9pZCc6IDAsICdjaGFuX2NudCc6IDAsICdjcmVhdGVfdGltZSc6IDM1MjIyNX1dfQ=='", "X-Code": 
        1000000, "X-Desc": "success", "X-IsSync": true}


         "X-Desc": "success"表示成功处理得到预测结果。
         只需要关注X-Server-Param这个字段，与header中的一个字段同名，但二者没有任何关系。这个字段的值是以b64格式返回的，其
         明文形式如下：
         {'sid': '37889655', 'result': [{'PROBABILITY': {'array': [0.0116160912, 0.9883839088],
          'values': [0.0116160912, 0.9883839088]}, 'PREDICTION': 1.0, 'amount': 0, 's3_flux': 0, 's2_flux': 0,
           's1_flux': 0, 'sc_1': 0, 'sc_2': 5, 'buka_cnt': 0, 'home_city': 591, 'imei_ischang': 0, 'model_desc_ischang': -5, 
           'ic_no_num': 20201106101340, 'imei_num': 591, 'call_called_num': 0, 'rec_cnt': 1, 'all_cnt': 202011060000000, 
           'brand_id': 0, 'chan_cnt': 0, 'create_time': 352225}]}
        明文中的众多字段中大多数是对请求数据的复现，只需要关注'PREDICTION': 1.0这个，表示是高危号码，如果值是0则代表不是高危号码。
        而'values': [0.0116160912, 0.9883839088]则表示不是/是高危号码的预测概率。
        """
        # 下面则是对预测结果的处理，取出预测概率和预测结果
        result = response['X-Server-Param'][1:]
        result = base64.b64decode(result)
        result = result.decode('utf-8')
        result = result.replace("\'", "\"")
        result = json.loads(result)
        result = result['result'][0]
        # 预测为非高危/高危的概率
        probability = result['PROBABILITY']['values']
        print("预测概率:", probability)
        # 判断是/否为高危号码，对应0/1
        result = result['PREDICTION']
        print("判定结果:", result)
    except Exception as error:
        print(error)


send()

