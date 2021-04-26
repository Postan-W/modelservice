import tornado.ioloop
import tornado.web
import json
print("这是在test上的测试")
print(True)
class hello(tornado.web.RequestHandler):
  def get(self):
    self.write('Hello,xiaorui.cc')
class add(tornado.web.RequestHandler):
  def post(self):
    print(self.request.body)
api_addr = "add"
api_addr2 = "hello"
application = tornado.web.Application([
tornado.web.url(api_addr if api_addr.startswith("/") else "/" + api_addr, add),
tornado.web.url(api_addr2 if api_addr2.startswith("/") else "/" + api_addr2, hello),
])
if __name__ == "__main__":
  application.listen(12345)
  tornado.ioloop.IOLoop.instance().start()
