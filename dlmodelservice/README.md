建议使用Github或Typora等支持Markdown语法的软件查看。或者在Pycharm中安装Markdown support插件。在view栏选择show preiview only。

### 1. 项目结构

| 文件名             | 作用                   | 描述                                               |
| ------------------ | ---------------------- | -------------------------------------------------- |
| download_models.py | 启动容器时下载模型     | 下载模型后解压到/models目录下，并删除模型压缩包    |
| flask_service.py   | 定义Flask的web接口     | 加载对应的模型对象，在接口中用以预测用户传来的数据 |
| models.py          | 定义的模型类供加载使用 | 实现各个模型预测和获取模型信息的方法               |
| local_utilities.py | 自定义工具             | 对预测图片的处理、序列化模型的解析等               |
| Dockerfile         | 即Docker镜像构建文件   | 选择合适的base镜像，安装所需的Python库等           |
| run.sh             | Linux的shell脚本文件   | 作为Dockerfile的ENTRYPOINT的执行目标               |

项目镜像启动容器后，执行ENTRYPOINT所指定的run.sh文件，该文件中执行了两个Python脚本，首先是下载模型，然后是加载模型，然后web服务正式启动。

- **download_models.py**

  **第一步下载：**环境变量XQUERY_ADDR用于拼接下载连接；环境变量MODEL_ID、MODEL_VERSION、GUID用于下载的请求参数

  **第二步解压：**目前默认的模型压缩包的目录结构是model/model，即真正的模型内容是放在model/model下面。模型被解压后放在/models下。其中H5、Pb、Onnx等模型只有一个文件，解压后移动到了/models下，例如：/models/test.h5；对于SavedModel、Ckpt等有多个文件的模型，采用的是直接解压到/models下，所以真正的模型文件是在/models/model/model下面，而不是上面直接在/models下。

- **flask_service.py**

  **第一步：**判断是什么模型，然后加载生成模型对象model

  **第二步：**接口接收到用户的请求数据后使用已加载好的model进行预测，返回预测结果

- **local_utilities.py**

  定义了一系列函数

  | 函数名称                | 功能描述                                                     |
  | ----------------------- | ------------------------------------------------------------ |
  | find_latest             | ckpt模型可能分为了多个global step，这里时选择最大编号即最新的加载 |
  | universal_image_process | 将服务接收的用户传来的图片转换为模型需要的输入形状，返回一个numpy数组作为模型输入值 |
  | h5_input_shape          | 通过正则表达式从H5模型的序列化信息中解析出模型输入           |
  | savemodel_input_shape   | 通过正则表达式从SavedModel模型的序列化信息中解析出模型输入(通过keras的方法保存的sm模型) |

- run.sh

  将传来的HOSTS环境变量写进hosts文件，定义了下载模型和启动服务

  

  ```
  string=$HOSTS
  array=${string//,/ }
  for var in ${array[@]} ;do echo ${var//<->/ } >> /etc/hosts ; done
  python /root/model_server/download_models.py
  #指定worker数、地址与端口、模块和flask对象、启动目录
  gunicorn -w 1 -b 0.0.0.0:5000 flask_service:app  --chdir /root/model_server/
  ```

- Dockerfile

  应该具备的操作大致有：一个合理的基础镜像、通过RUN安装项目所需的各种库(下面将给出)、使用COPY或者ADD指令将项目放在某个位置、EXPOSE服务的5000端口、ENTRYPOINT执行run.sh脚本。

  目前在140.210.90.56机器上(root/cvlab20200914!),，有正在使用的完善的镜像，都是以1135085247/modelservice_tf2_cpu命名的，tag越大表示越新。所以当修改了项目代码，只需把最新的的镜像作为基础镜像，然后将项目代码COPY到基础镜像的项目代码位置(也就是覆盖)即可，一般不用执行其他操作。

## **2.本地启动容器示例**

非GPU版，不带--gpus参数

docker run -ti --name localtest -p 5000:5000 --env-file 环境变量文件 -v 本地路径:容器路径  镜像名称  bash

## 3.Python库

Package              Version
-------------------- ---------------
absl-py              0.10.0
aiohttp              3.6.2
asn1crypto           0.24.0
astor                0.8.1
async-timeout        3.0.1
attrs                20.2.0
cachetools           4.1.1
certifi              2020.6.20
chardet              3.0.4
click                7.1.2
cryptography         2.1.4
cycler               0.10.0
fire                 0.3.1
Flask                1.1.2
future               0.18.2
gast                 0.2.2
google-auth          1.22.0
google-auth-oauthlib 0.4.1
google-pasta         0.2.0
grpcio               1.32.0
gunicorn             20.0.4
h5py                 2.10.0
idna                 2.6
idna-ssl             1.1.0
importlib-metadata   2.0.0
itsdangerous         1.1.0
Jinja2               2.11.2
joblib               1.0.0
Keras                2.3.1
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.2
keras2onnx           1.7.0
keyring              10.6.0
keyrings.alt         3.0
kiwisolver           1.3.1
Markdown             3.2.2
MarkupSafe           1.1.1
matplotlib           3.3.4
multidict            4.7.6
numpy                1.18.5
oauthlib             3.1.0
onnx                 1.8.0
onnxconverter-common 1.7.0
onnxruntime          1.6.0
opencv-python        4.5.1.48
opt-einsum           3.3.0
pandas               1.1.5
Pillow               8.1.0
pip                  21.0.1
protobuf             3.13.0
py4j                 0.10.9
pyasn1               0.4.8
pyasn1-modules       0.2.8
pycrypto             2.6.1
pygobject            3.26.1
pyparsing            2.4.7
pypmml               0.9.9
pyspark              3.0.1
python-apt           1.6.5+ubuntu0.3
python-dateutil      2.8.1
pytz                 2020.5
pyxdg                0.25
PyYAML               5.3.1
requests             2.24.0
requests-oauthlib    1.3.0
rsa                  4.6
scikit-learn         0.24.1
scipy                1.5.4
SecretStorage        2.3.1
setuptools           54.0.0
six                  1.11.0
sklearn              0.0
sklearn-pandas       2.0.4
sklearn2pmml         0.66.1
tensorboard          2.0.2
tensorflow           2.0.0
tensorflow-estimator 2.0.1
termcolor            1.1.0
threadpoolctl        2.1.0
torch                1.5.0+cpu
torchvision          0.6.0+cpu
typing-extensions    3.7.4.3
urllib3              1.25.10
Werkzeug             1.0.1
wheel                0.30.0
wrapt                1.12.1
yarl                 1.6.0
zipp                 3.2.0