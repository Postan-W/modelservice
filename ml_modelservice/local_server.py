from flask import Flask, jsonify, request
import json
app = Flask(__name__)

@app.route('/get',methods=['GET'])
def receive_get_json():
    request_args = request.args
    print("请求的参数类型是：",type(request_args))
    print("请求的参数是:",request_args)
    request_args = dict(request_args)
    return request_args

app.run(host="0.0.0.0", port=5000, debug=False)