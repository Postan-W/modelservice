lis1 = [{'节点': 'mn1', '更新时间': '2021/2/26  11:10:02'}]
print(str(lis1))

from zipfile import ZipFile

with ZipFile("modelsm.zip",'r') as f:
    print(list(filter(lambda x: x.startswith('model/model'), f.namelist())))
    zips = list(filter(lambda x: x.startswith('model/model'), f.namelist()))
    f.extractall("./models",zips)
