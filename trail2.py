from request_json import data,data2,data5,data3
import base64
import tensorflow as tf


def universal_image_process(image: bytes, shape: list):
 print("tag:universal_image_process正在处理图片,将图片转为numpy数组")
 image = Image.open(io.BytesIO(image))
 try:
  len_shape = len(shape)
  image_array = np.array(image)
  # 如果输入张量是一维的，那么需要把图片拉平为一维的
  if len_shape == 1:
   # 如果图片是3通道的RGB图片
   if len(image_array.shape) == 3:
    print("3通道图片处理成1维张量")
    # 把图片resize到shape要求的尺寸大小
    quotient = int(shape[0] / 3)  # 这个quotient*3比shape[0]少的部分后面补齐
    image2 = image.resize((quotient, 1))
    image2 = np.array(image2)  # image2.shape->(1,quotient,3)
    product = 1
    for i in range(3):
     product *= image2.shape[i]
    image2.shape = (product,)
    image2 = list(image2)
    # 把不够的部分补上
    for i in range(shape[0] - product):
     image2.append(127)  # 取一个中间值补上
    image2 = np.array(image2, dtype="float32")
    image2 /= 255.0
    image2.shape = (1,) + image2.shape
    return image2
   elif len(image_array.shape) == 2:
    print("1通道处理成一维张量")
    image3 = image.resize((shape[0], 1))
    image3 = np.array(image3, dtype="float32")
    image3 /= 255.0
    return image3
  elif len_shape == 2:
   if len(image_array.shape) == 3:
    print("把3通道的处理成二维张量")
    image4 = image.resize((int(shape[0] * shape[1] / 3), 1))
    image4 = np.array(image4)
    image4.shape = (int(shape[0] * shape[1] / 3) * 3,)
    image4 = list(image4)
    for i in range(shape[0] * shape[1] - int(shape[0] * shape[1] / 3) * 3):
     image4.append(127)
    image4 = np.array(image4, dtype="float32")
    image4.shape = (1,) + tuple(shape)
    image4 /= 255.0
    return image4
   elif len(image_array.shape) == 2:
    print("把单通道的处理成二维张量")
    image5 = image.resize((shape[1], shape[0]))
    image5 = np.array(image5, dtype="float32")
    print("图片形状为:", image5.shape)
    image5 /= 255.0
    image5.shape = (1,) + tuple(shape)
    return image5
  elif len_shape == 3:
    if len(image_array.shape) == 3:
     print("把3通道的转为3维张量")
     image6 = image.resize((shape[1], shape[0]))
     image6 = np.array(image6, dtype="float32")
     image6 /= 255.0
     image6.shape = (1,) + tuple(shape)
     return image6
    if len(image_array.shape) == 2:
     print("把单通道的转为3维张量")
     if shape[2] == 1:
      print("输入张量把单通道也在最后一个维度上体现了")
      image7 = image.resize((shape[1], shape[0]))
      image7 = np.array(image7, dtype="float32")
      image7 /= 255.0
      image7.shape = (1,) + tuple(shape)
     else:
      image7 = image.convert("RGB")
      image7 = image7.resize((shape[1], shape[0]))
      image7 = np.array(image7, dtype="float32")
      image7 /= 255.0
      image7.shape = (1,) + tuple(shape)
     return image7
 except Exception as e:
  print(e)


img_b64 = data["b64"]
img_b64 = base64.b64decode(img_b64)
with tf.compat.v1.Session() as sess:
 img_b64 = tf.compat.v1.decode_raw(img_b64,tf.uint8)
 print(type(img_b64))
 img_b64 = tf.compat.v1.reshape(img_b64, [1,50926,1,1])
 img_b64 = tf.cast(img_b64, tf.float32) * 1.0 / 255
 print(type(img_b64))
 img_b64 = sess.run(img_b64)
 print("转化后的输入类型是:",type(img_b64),"形状是:",img_b64.shape)

from PIL import Image
import numpy as np
import io
image = data5["b64"]
image = base64.b64decode(image)
# image = Image.open(io.BytesIO(image))
# image = np.array(image)
# print(image.shape)
image = universal_image_process(image,shape=[32,32,1])
print(image.shape)