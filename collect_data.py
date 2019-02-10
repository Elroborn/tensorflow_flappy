"""
Created on 2019/2/9 18:00
@File: collect_data.py
@author: coderwangson
"""
"#codeing=utf-8"
import cv2 as cv

capture = cv.VideoCapture(0)
count = 100
for i in range(count):
    _,img = capture.read()
    img = cv.flip(img,1)
    img = img[50:350,426:640,:]
    cv.imwrite("./bu/"+"1"+str(i)+".jpg",img)
    cv.imshow("1",img)
    print(i)
    cv.waitKey(24)