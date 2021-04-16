# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:03:45 2021

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:46:46 2020

@author: Administrator
"""




#from collections import Counter
#import data_science
#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import IsolationForest
import pandas as pd
#import numpy as np
import datetime
#import os
t1=datetime.datetime.now()
#-------------------参数设置---------------------
data_path='中台HIVE任务近一个月连接数.xlsx'
node_col='节点'#节点列名
ts_col='更新时间'#时间列名
x_col='连接数'#入模特征
method='三倍标准差'#异常临界值计算方法有'三倍标准差'和'箱型图'两种

#-------------------读取数据---------------------
if 'csv' in data_path[-4:]:
    data_ori=pd.read_csv(data_path)
else:
    data_ori=pd.read_excel(data_path)
#data_ori = data_science.Dataset("vbap4272275901a94ef1d5095b14").get_dataframe()

#-------------------数据处理---------------------
#去重
data_ori=data_ori.drop_duplicates([node_col,ts_col])

#空值处理--删除
data_ori=data_ori[[node_col,ts_col,x_col]]
data_ori=data_ori.dropna(how='any')

#提取星期、小时
data_ori=data_ori.rename(columns={ts_col:'datetime'})
data_ori['datetime']=pd.to_datetime(data_ori['datetime'])
data_ori['weekday']=data_ori['datetime'].apply(lambda x:x.weekday()+1)#提取星期
data_ori['is_work']=data_ori['weekday'].apply(lambda x:1 if x<6 else 0)#是否工作日，1是0否
data_ori['hour']=data_ori['datetime'].apply(lambda x:x.hour+2 if x.hour%2==0 else x.hour+1)
#data_ori['hour_']=data_ori['datetime'].apply(lambda x:x.hour+round(x.minute/60,2))

#-------------------模型建立---------------------
#提取不同数据库不同小时数据进行异常检测
node_list=data_ori[node_col].unique()
iswork_list=sorted(data_ori['is_work'].unique())
hour_list=sorted(data_ori['hour'].unique())
result=[]
for nd in node_list:

    for iw in iswork_list:
        for ho in hour_list:
            print('----------------正在检测节点{}_是否工作日{}_时间段{}-{}时-------------------'.format(nd,iw,ho-1,ho))
            data=data_ori.loc[(data_ori[node_col]==nd)&(data_ori['is_work']==iw)&(data_ori['hour']==ho)]
            
            if len(data)>0: 
#                data=data.sort_values(['datetime'])  
                if method=='三倍标准差':
                    #三倍标准差
                    data_input= data[x_col]
                    mean=data_input.mean()#计算均值
                    std=data_input.std()#计算标准差
                    lower=mean-3*std#计算下限
                    upper=mean+3*std#计算上限
                
                else:
                     #箱型图
                    data_input= data[x_col]
                    quantile25=data_input.quantile(0.25)#计算下四分位数
                    quantile75=data_input.quantile(0.75)#计算上四分位数
                    iqr=quantile75-quantile25#计算四分位距
                    lower=quantile25-1.5*iqr#计算下限
                    upper=quantile75+3*iqr#计算上限                
                result.append([nd,iw,ho,upper])
            else:
                result.append([nd,iw,ho,None])

result_df=pd.DataFrame(result,columns=['节点','是否工作日','时间段','异常临界值']) 
result_df['计算时间']=t1

#存储结果
with pd.ExcelWriter('异常临界值.xlsx') as writer:
    result_df.to_excel(writer,sheet_name=method,index=False)

#data_science.Dataset("vbap37c90164eba01c0121f73354").write_with_dataframe(result_df)

t2=datetime.datetime.now()
print('finish! spend ts--{}'.format(t2-t1))
