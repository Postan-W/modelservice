from flask import Flask,send_from_directory,request
import requests
import json

app = Flask(__name__)

@app.route('/download',methods=["GET","POST"])
def downloadzip():
    return send_from_directory("C:\\Users\\15216\\Desktop", filename="model.zip", as_attachment=False)

@app.route('/addr/dsModel/downloadModelForModelService',methods=['GET','POST'])
def downloadzip2():
    if request.method == "GET":
        print("GET请求")
        args = request.args.to_dict()
        print("所有请求参数是:", args)
        print("请求参数的类型是：",type(args))
        with open("modelonnx.zip", "rb") as f:
            content = f.read()
            print("modelonnx.zip读取成功")
            return content
    elif request.method == "POST":
        print("POST请求")
        request_json = json.loads(request.data)
        print("post请求发来的json是：",request_json)
        with open("modelonnx.zip", "rb") as f:
            content = f.read()
            print("modelonnx.zip读取成功")
            return content

app.run("0.0.0.0",6543)
