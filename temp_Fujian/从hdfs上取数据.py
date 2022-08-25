"""
@Time : 2021/4/15 15:13
@Author : wmingzhu
@Annotation : 
"""
from hdfs.client import Client,InsecureClient
import os
hdfshost = "http://localhost:50070"
csv_path = "/wmz1/"
local_file_path = "./hdfs_files"
def do():
    global csv_path
    client = Client(hdfshost)
    file_list = client.list(csv_path)
    print(file_list)
    for file in file_list:
        if file.endswith(".csv"):
            csv_path = csv_path + file
    # 读取csv并同名写到本地
    with open("./异常临界值local.csv", 'w', encoding='GB2312') as local:
        with client.read(csv_path, encoding='GB2312') as hdfs:
            for line in hdfs:
                local.write(line.strip('\n'))
                # local.write(line)




def do2():
    global csv_path
    client = Client(hdfshost)
    client.download(csv_path,local_file_path)

do2()
