# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:24:03 2021

@author: Administrator
"""
"""
docker run -ti --name aiops --net host --env hdfsHost=hdfs://fjedcprohd --env csvPath=/user/bdoc/24/services/hive/ods_prod/ai_data/b252b926f9c647409b2694c07b71f9f3/dataSet/vbape740573f913677258af3aff4/data
"""
"""
请求示例：curl -H "Content-Type:application/json" -X POST -d '{"data":[{'节点':'mn1','更新时间':'2021/2/26  11:10:02','连接数':'100'},
             {'节点':'mn2','更新时间':'2021/2/26  11:10:02','连接数':'80'},
             {'节点':'mn3','更新时间':'2021/2/26  11:10:02','连接数':'80'}]}' 'http://localhost:5000/anomalyDetection'
"""
# -*- coding: utf-8 -*-
import pandas as pd
#import json

def gethour(x):
    '''建立时间段解析函数'''
    if x%2==0:
        return  x+2 
    else:
        return x+1
    
def detect(input_param,data_limit):
    '''异常检测'''
    result=[]#异常检测结果
    for data in input_param:
        node=data['节点']
        iswork=int(pd.to_datetime(data['更新时间']).weekday()<5)#解释是否工作日，weekday()的值是0-6,0-4代表工作日
        hour=gethour(pd.to_datetime(data['更新时间']).hour)#解析时间段
        limit=data_limit.query("column_0 in{} & column_1=={} & column_2=={}".format([node],iswork,hour))['column_3']#提取异常临界值
        if len(limit)>0:
            limit=list(limit)[0]#这里仅以第一个为标准有待考察临界值文件的生成代码
            if int(data['连接数'])>limit:
                data['是否异常']='是'
            else:
                data['是否异常']='否' 
        else:
            data['是否异常']='无法检测'#没有临界值则无法检测
        result.append(data)
    return result



if __name__=='__main__':
    input_param=[{'节点':'mn1','更新时间':'2021/2/26  11:10:02','连接数':'100'},
             {'节点':'mn2','更新时间':'2021/2/26  11:10:02','连接数':'80'},
             {'节点':'mn3','更新时间':'2021/2/26  11:10:02','连接数':'80'}]#输入参数

    # data_limit=pd.read_excel('异常临界值.xlsx',engine='openpyxl')#读取已经计算好的临界值数据
    try:
        data_limit = pd.read_csv('异常临界值local.csv', encoding='utf-8')
    except:
        print("不是utf-8编码")
        data_limit = pd.read_csv('异常临界值local.csv', encoding='GB2312')
    if type(input_param)==list:
        result=detect(input_param,data_limit)#异常检测
        print('success!')
        print(result)
    elif type(input_param)==dict:
        result=detect([input_param],data_limit)#异常检测
        print('success!')
        print(result)
    else:
        print('输入格式不对！')
        