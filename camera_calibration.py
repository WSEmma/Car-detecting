import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import pylab

cal = plt.imread('calibration.jpg')
fig = plt.figure(1,figsize=(16,9))  #创建画布
cal_gray = cv2.cvtColor(cal,cv2.COLOR_RGB2GRAY) #获取灰度图像

#左右分别显示原图与其灰度图
plt.subplot(2,2,1)
plt.imshow(cal)
plt.subplot(2,2,2)
plt.imshow(cal_gray,cmap='gray')

ret,corners = cv2.findChessboardCorners(cal_gray,(6,4),None)    #寻找棋盘上的内角点
if ret == True:
    cal = cv2.drawChessboardCorners(cal,(6,4),corners,ret)  #绘制内角点
plt.imshow(cal)

#获取内角点在现实世界中所在的相对位置
objp = np.zeros((4*6,3),np.float32) #共有4*6个内角点，用三维坐标表示
objp[:,:2] = np.mgrid[0:6,0:4].T.reshape(-1,2)  #利用mgrid转置矩阵构造网格二维坐标

img_points = []
obj_points = []

img_points.append(corners)
obj_points.append(objp)

image_size = (cal.shape[1],cal.shape[0])    #图像大小
ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(obj_points,img_points,image_size,None,None)
print(mtx)

#矫正图像
img = cv2.imread('calibration.jpg')
undist = cv2.undistort(img,mtx,dist,None,mtx)
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(undist)
pylab.show()