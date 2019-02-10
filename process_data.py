"""
Created on 2019/2/9 18:06
@File: process_data.py
@author: coderwangson
"""
"#codeing=utf-8"
import os
import cv2 as cv
import numpy as np
import pickle
def process():
    dirname = "./bu/"
    a = []
    for f in os.listdir(dirname):
        img = cv.imread(dirname+f)
        img = cv.resize(img,(64,64))
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.reshape(img,(64,64,1))
        a.append(img)
    a = np.array(a)
    with open("bu.pkl","wb") as f:
        pickle.dump(a,f)

def get_data():
    shitou = None
    shitou_label = None
    bu = None
    bu_label = None
    with open("shitou.pkl","rb") as f:
        shitou = pickle.load(f)
        shitou_label = np.zeros((shitou.shape[0],1))

    with open("bu.pkl", "rb") as f:
        bu = pickle.load(f)
        bu_label = np.ones((bu.shape[0],1))
    data = np.vstack([shitou,bu])
    label = np.vstack([shitou_label,bu_label])
    label = label.astype(int)
    label = np.eye(2)[label].reshape(data.shape[0],2)
    return data,label
def get_batch(batch_size):
    data,label = get_data()
    p = np.random.permutation(data.shape[0])[0:batch_size]
    return data[p],label[p]
