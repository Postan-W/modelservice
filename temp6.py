#coding=utf-8
import json
from temp2 import data,data3
import requests
import json
argus = {"name":"wmingzhu","age":30,"location":"beijing","gender":"male"}
# res = requests.get("http://127.0.0.1:5000/download",params={"name":"wmingzhu","age":30,"location":"beijing","gender":"male"})
#res = requests.post("http://127.0.0.1:5000/service",data=json.dumps(data)).text
res_info = requests.get("http://127.0.0.1:5000/service/metadata").text
res2 = requests.post("http://127.0.0.1:5000/service",data=json.dumps(data3)).text
#print("type=b64的模型输出结果为：",res)
print("模型的信息为:",res_info)
print("type=url的模型输出结果为:",res2)


