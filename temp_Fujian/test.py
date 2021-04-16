"""
@Time : 2021/4/15 17:39
@Author : wmingzhu
@Annotation : 
"""
# import requests
# # import json
# # data = {"data":[{'节点':'mn1','更新时间':'2021/2/26  11:10:02','连接数':'100'},
# #              {'节点':'mn2','更新时间':'2021/2/26  11:10:02','连接数':'80'},
# #              {'节点':'mn3','更新时间':'2021/2/26  11:10:02','连接数':'80'}]}
# # result = requests.post("http://127.0.0.1:5000/anomalyDetection",data=json.dumps(data)).text
# # print(result)

import os
csv_path = "./"
def do():
    global csv_path
    file_list = os.listdir(csv_path)
    for file in file_list:
        if file.endswith(".csv"):
            csv_path = csv_path + file
            break

do()
print("完整路径是:",csv_path)

