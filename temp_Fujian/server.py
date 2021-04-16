"""
@Time : 2021/4/15 16:00
@Author : wmingzhu
@Annotation : 归属项目：福建数据科学平台。功能：节点连接数异常检测
"""
from flask import Flask,jsonify,request
import logging
from hdfs.client import Client
import os
import json
import pandas as pd
from data_process import detect
#测试
# os.environ["hdfsHost"] = "http://localhost:50070"
# os.environ["csvPath"] = "/wmz1/异常临界值.csv"
#hdfs主机
hdfshost = os.environ.get("hdfsHost")
#异常临界值的csv文件路径,即hdfs上的路径
csv_path = os.environ.get("csvPath")
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s--文件路径:%(pathname)s--行号:%(lineno)s--日志内容:%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
print(hdfshost,csv_path)

@app.route('/anomalyDetection', methods=['POST'])
def anomalyDetection():
    try:
        global csv_path
        hdfs_client = Client(hdfshost)
        file_list = hdfs_client.list(csv_path)
        for file in file_list:
            if file.endswith(".csv"):
                csv_path = csv_path +file
                break

    except Exception as e:
        app.logger.info("连接hdfs出现错误")
        app.logger.info(e)
    # 读取csv并同名写到本地
    try:
        with open("./local.csv", 'w', encoding='GB2312') as local:
            with hdfs_client.read(csv_path, encoding='GB2312') as hdfs:
                for line in hdfs:
                    # local.write(line.strip('\n'))#不去换行也行，不影响数据分析，还省得出错
                    local.write(line)
    except Exception as e:
        app.logger.info("获取hdfs文件失败")
        app.logger.info(e)

    #开始执行分析
    data = json.loads(request.data)["data"]

    try:
        data_limit = pd.read_csv('local.csv', encoding='GB2312')
    except Exception as e:
        app.logger.info(e)

    try:
        result = detect(data, data_limit)
    except Exception as e:
        app.logger.info(e)

    return str(result).encode('utf-8')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)