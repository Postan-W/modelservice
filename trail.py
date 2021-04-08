from models import H5Model,SMModelTf2,SavedModelFromFlow
from request_json import data,data2,data3,data5,data6

# model = H5Model("./model/miniVGGNet/model/model/model_miniVGGNet.h5")
#
# model.predict(data3)
#
# print("下面是MobileNet_V1")
#
# model2 = H5Model("./model/MobileNet_V1/model/model/model_MobileNet_v1.h5")
#
# model2.predict(data3)
# print("下面是LeNet")
# model3 = SMModelTf2("./model/LeNet/model/model/")

#下面定义tf1版本的SavedModel模型加载代码
# def method1():
#     with tf.Session() as sess:
#         meta_graph = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],model_path)#被加载的图就是当前的默认图
#         #从下面的代码形式中可以猜测meta_graph.signature_def["xxx"]返回的是一个对象
#         input_tensor_name = meta_graph.signature_def["graph_information"].inputs["model_input"].name
#         output_tensor_name = meta_graph.signature_def["graph_information"].outputs["model_output"].name
#
#         graph = tf.get_default_graph()
#
#         input_tensor = graph.get_tensor_by_name(input_tensor_name)
#         output_tensor = graph.get_tensor_by_name(output_tensor_name)
#
#         print(sess.run(output_tensor,feed_dict={input_tensor: 20}))
import tensorflow.compat.v1 as tf
print(tf.__version__)

t1_saved_model_path = "./model/LeNet/model/model/"
# with tf.Session() as sess:
#     meta_graph = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],t1_saved_model_path)
#     signature_def = meta_graph.signature_def
#     #其中流上训练的模型使用的签名的key叫做test_signature,然后inputs值的input_x键对应输入张量
#     signature_def =signature_def["test_signature"]
#     input_tensor_name =signature_def.inputs['input_x'].name
#     input_tensor = tf.get_default_graph().get_tensor_by_name(input_tensor_name)
#     #获取输入张量的形状
#     input_shape = input_tensor.shape
#     print(input_shape)
#     print(input_tensor_name)

model4 = SavedModelFromFlow(t1_saved_model_path)

model4.predict(data6)

