import json
import requests
import base64

str = """{"appid":"45665e33","uid":"3568732","csid":"37889655","stream_id":"image-1","frame_id":1,"last_frame":"true","prediction_data":{"amount": 0,"s3_flux": 0,"s2_flux":0,"s1_flux":0,"sc_1":0,"sc_2":5,"buka_cnt":0,"home_city":591,"imei_ischang":0,"model_desc_ischang":-5,"ic_no_num":20201106101340,"imei_num":591,"call_called_num":0,
    "rec_cnt":1,"all_cnt":202011060000000,"brand_id":0,"chan_cnt":0,"create_time":352225}}""".encode('utf-8')

en = base64.b64encode(str)
data = {"X-Server-Param":"eyJhcHBpZCI6IjQ1NjY1ZTMzIiwidWlkIjoiMzU2ODczMiIsImNzaWQiOiIzNzg4OTY1NSIsInN0cmVhbV9pZCI6ImltYWdlLTEiLCJmcmFtZV9pZCI6MSwibGFzdF9mcmFtZSI6InRydWUiLCJwcmVkaWN0aW9uX2RhdGEiOnsiYW1vdW50IjogMCwiczNfZmx1eCI6IDAsInMyX2ZsdXgiOjAsInMxX2ZsdXgiOjAsInNjXzEiOjAsInNjXzIiOjUsImJ1a2FfY250IjowLCJob21lX2NpdHkiOjU5MSwiaW1laV9pc2NoYW5nIjowLCJtb2RlbF9kZXNjX2lzY2hhbmciOi01LCJpY19ub19udW0iOjIwMjAxMTA2MTAxMzQwLCJpbWVpX251bSI6NTkxLCJjYWxsX2NhbGxlZF9udW0iOjAsCiAgICAicmVjX2NudCI6MSwiYWxsX2NudCI6MjAyMDExMDYwMDAwMDAwLCJicmFuZF9pZCI6MCwiY2hhbl9jbnQiOjAsImNyZWF0ZV90aW1lIjozNTIyMjV9fQ==","X-CurTime":1502184180,"X-CheckSum":"001388491350e266fab5e15da9aea749","X-Param": {"group_name":"11211"}}

res2 = requests.post("http://localhost:5000/v1/request",data=json.dumps(data)).text
print("预测结果是:",res2)