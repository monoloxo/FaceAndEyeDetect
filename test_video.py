# encoding:utf-8
import cv2
import numpy as np

# 运行之前，检查cascade文件路径是否在相应的目录下
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 读取图像
##img =cv2.imread('img/timg.jpg')
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cap = cv2.VideoCapture('rtsp://admin:admin123@192.168.199.67:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif')
while (True):
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测脸部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),        	flags=cv2.CASCADE_SCALE_IMAGE)
    print('Detected ', len(faces), " face")

    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
#cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

