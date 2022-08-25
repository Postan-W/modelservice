"""
@Time : 2021/4/15 16:00
@Author : wmingzhu
@Annotation : 归属项目：福建数据科学平台。功能：节点连接数异常检测
"""
#在centos中yum install krb5-libs krb5-server krb5-workstation即可使用Kerberos客户端
#krb5文件放在/etc下面
#直接在/etc/hosts里修改，启动容器因为容器机制的原因修改的内容会丢掉，而不是在镜像中添加,进入容器后将这里hosts2.txt的内容添加到/etc/hosts文件中
#没有entrypoint，需要后台启动-td容器，然后exec进去，然后手动python /root/service/server.py

"""
docker run -ti --name localtest --net host anomalydetection:ultimate bash
"""
"""
curl -H "Content-Type:application/json" -X POST -d '{"data":[{"节点":"mn1","更新时间":"2021/2/26  11:10:02","连接数":"100"}, {"节点":"mn2","更新时间":"2021/2/26  11:10:02","连接数":"80"}, {"节点":"mn3","更新时间":"2021/2/26  11:10:02","连接数":"80"}]}' 'http://localhost:5000/anomalyDetection'
"""
from flask import Flask,jsonify,request
import logging
from hdfs.client import Client
import os
import json
import pandas as pd
from data_process import detect
from hdfs.ext.kerberos import KerberosClient
from krbcontext import krbContext

csv_path = "/user/bdoc/24/services/hive/ods_prod/ai_data/b252b926f9c647409b2694c07b71f9f3/dataSet/vbape740573f913677258af3aff4/data/vbape740573f913677258af3aff4_0.csv"
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s--文件路径:%(pathname)s--行号:%(lineno)s--日志内容:%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


@app.route('/anomalyDetection', methods=['POST'])
def anomalyDetection():

    global csv_path
    #客户端认证
    os.system("kinit -kt renter_dp_prod.keytab renter_dp_prod/bdoc@FJBCHKDC")
    with krbContext(using_keytab=True, keytab_file="renter_dp_prod.keytab",
                    principal="renter_dp_prod/bdoc@FJBCHKDC"):
        client = KerberosClient("http://10.48.138.244:50070")
        client.download(csv_path,"local.csv",overwrite=True)

    #开始执行分析
    data = json.loads(request.data)["data"]


    data_limit = pd.read_csv('local.csv', encoding='GB2312')


    result = detect(data, data_limit)


    return str(result).encode('utf-8')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=False)