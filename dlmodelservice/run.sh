echo "将HOSTS环境变量加入hosts文件"
source /etc/profile
string=$HOSTS
array=${string//,/ }
for var in ${array[@]} ;do echo ${var//<->/ } >> /etc/hosts ; done
python /root/model_server/download_models.py
#指定worker数、地址与端口、模块和flask对象、启动目录
gunicorn -w 1 -b 0.0.0.0:5000 flask_service:app  --chdir /root/model_server/
