"""
Created on 2019/2/9 19:16
@File:predict.py
@author: coderwangson
"""
"#codeing=utf-8"
import tensorflow as tf
import numpy as np
import cv2 as cv
import time
def predict(img,sess,x_image,y_,keep_drop):
    y = sess.run(y_, feed_dict={x_image:img, keep_drop: 1.0})
    return np.argmax(y)
# 因为sess加载耗时间，所以在程序中使用，这样我们就能避免每次加载耗时间
# start = time.clock()
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph("./model/hand.meta")
#     saver.restore(sess,tf.train.latest_checkpoint("./model"))
#     graph = tf.get_default_graph()
#     x_image = graph.get_tensor_by_name("Placeholder:0")
#     y_ = graph.get_tensor_by_name("add_3:0")
#     keep_drop = graph.get_tensor_by_name("Placeholder_2:0")
#     # print(time.clock() - start)
#     capture = cv.VideoCapture(0)
#     while True:
#         ha, img = capture.read()
#         img = cv.flip(img,1)
#         cv.rectangle(img, (426, 50), (640, 350), (170, 170, 0))
#         img = img[50:350, 426:640, :]
#         img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         cv.imshow("img", img)
#         img = cv.resize(img, (64, 64)).reshape((1, 64, 64, 1))
#
#         # 0是拳头 1 是手掌
#         action = predict(img, sess,x_image,y_,keep_drop)
#         if action ==1:
#             print("张开")
#         elif action==0:
#             print("合住")
#         cv.waitKey(24)
