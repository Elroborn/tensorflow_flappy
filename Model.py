"""
Created on 2019/2/9 18:26

@author: coderwangson
"""
"#codeing=utf-8"
import tensorflow as tf
from process_data import get_batch
x_image = tf.placeholder(tf.float32,[None,64,64,1])

y_ = tf.placeholder(tf.float32,[None,2])

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape = shape))
# 关于same和valid
# https://blog.csdn.net/wuzqchom/article/details/74785643
def conv_2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding ="SAME")
def max_pool(x):
    return tf.nn.max_pool(x,ksize =[1,2,2,1],strides =[1,2,2,1],padding ="SAME")

w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv_2d(x_image,w_conv1)+b_conv1)
h_pool1 = max_pool(h_conv1)

# 第二个卷积操作
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv_2d(h_pool1,w_conv2)+b_conv2)
h_pool2 = max_pool(h_conv2)
#开始全连接
w_fc1 = weight_variable([16*16*64,1024])
b_fc1 = weight_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,16*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
keep_drop = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_drop)
w_fc2 = weight_variable([1024,2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop,w_fc2)+b_fc2
print(y_conv)
#  tf.nn.sofmax_cross_ entropy_ with_logits 函数 ， 它可以直接对
# Logit 定义交叉烟损失 ， 写法为
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels =y_,
                                                                       logits = y_conv ))

train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct = tf.equal(tf.argmax(y_,axis = 1),tf.argmax(y_conv,axis = 1))
acc = tf.reduce_mean(tf.cast(correct,tf.float32))



saver = tf.train.Saver()
ck_path ="./model/hand"
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        batch = get_batch(50)
        if i %10 ==0:
            # 原来我写的是acc = sess.run(acc,feed_dict = {x:batch[0],y_:batch[1],keep_drop:1.0})
            # 这样acc会变成常数导致出错
            acc_v = sess.run(acc,feed_dict = {x_image:batch[0],y_:batch[1],keep_drop:1.0})
            print(acc_v)
        if i%10 ==0:
            print("save")
            saver.save(sess,ck_path,write_meta_graph=True)
        sess.run(train_op,feed_dict = {x_image:batch[0],y_:batch[1],keep_drop:1.0})
