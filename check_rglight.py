#encoding:utf8
import cv2
import cv2 as cv
import time
import numpy as np
import math


#查找左边界
def findLeft(pt,Tg1,Tg2,gray):
    m = pt[0] - 1
    if m >= 0:
        gs = gray[pt[1]][m]
        point = (m,pt[1])
        if(gs < Tg1):
            m = findLeft(point,Tg1,Tg2,gray)
        elif(gs < Tg2):
            m = findLeft(point,Tg1,Tg2,gray)
        else:
            m = pt[0]
    else:
        m = pt[0]
    return m

#查找上边界
def findTop(pt,Tg1,Tg2,gray):
    n = pt[1] - 1
    if n >= 0:
        gs = gray[n][pt[0]]
        point = (pt[0],n)
        if(gs < Tg1):
            n = findTop(point,Tg1,Tg2,gray)
        elif(gs < Tg2):
            n= findTop(point,Tg1,Tg2,gray)
        else:
            n = pt[1]
    else:
        n = pt[1]
    return n
     
#查找右边界
def findRight(pt,Tg1,Tg2,gray,line):
    m = pt[0] + 1
    if m <= line - 1:
        gs = gray[pt[1]][m]
        point = (m,pt[1])
        if(gs < Tg1):
            m = findRight(point,Tg1,Tg2,gray,line)
        elif(gs < Tg2):
            m = findRight(point,Tg1,Tg2,gray,line)
        else:
            m = pt[0]
    else:
        m = pt[0]
    return m

#查找下边界
def findBottom(pt,Tg1,Tg2,gray,sample):
    n = pt[1] + 1
    if n <= sample - 1:
        gs = gray[n][pt[0]]
        point = (pt[0],n)
        if(gs < Tg1):
            n = findBottom(point,Tg1,Tg2,gray,sample)
        elif(gs < Tg2):
            n= findBottom(point,Tg1,Tg2,gray,sample)
        else:
            n = pt[1]
    else:
        n = pt[1]
    return n

#查找红绿灯黑色边框
def findPoint(pt1,pt2,Tg1,Tg2,line,sample,gray):
    left = findLeft(pt1,Tg1,Tg2,gray)
    right = findRight(pt2,Tg1,Tg2,gray,line)
    top = findTop(pt1,Tg1,Tg2,gray)
    bottom = findBottom(pt2,Tg1,Tg2,gray,sample)
#     if right - left >= 5 and right - left <= 30 and bottom - top >= 5 and bottom - top <= 30:
    return ((left,top),(right,bottom))
#     else:
#         return None


#对文件进行处理
def imageProc(gray,gray1,src,T1,T2,T3,T4,T5,T6,T7,T8,T9,Tg1,Tg2):

    
     
    #寻找高亮区域轮廓
    ret,contours,heirs = cv2.findContours(gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        rect = cv2.boundingRect(contour)
#                     P = rect[2]/rect[3]
        P = ((rect[2] < rect[3]) and (rect[2]*1.0/rect[3]) or (rect[3]*1.0/rect[2]))
        S = rect[2]*rect[3]
        
        #左上角
        point1 = (rect[0],rect[1])
        #右下角
        point2 = ((rect[0]+rect[2]),(rect[1]+rect[3]))
        #左下角
        point3 = (rect[0],(rect[1]+rect[3]))
        #右上角
        point4 = ((rect[0]+rect[2]),rect[1])
#         cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))
        pcenter1 = (rect[0] + rect[2]/2,rect[1] + rect[3]/2)
        if P >= T2 and P <= T3 and S > T4 and S < T5:
            cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))
            fontx=cv2.putText(src,'red light',(rect[0]+rect[2],rect[1]+rect[3]),cv.FONT_HERSHEY_PLAIN ,1, (0,0,255), 1)
            pp = findPoint(point1,point2,Tg1,Tg2,gray1.shape[1],gray1.shape[0],gray1)
            
#                         if(pp != None):
            #黑色矩形框面积
            area = math.fabs(pp[0][0] - pp[1][0]) * math.fabs(pp[0][1] - pp[1][1])
            w1 = math.fabs(pp[0][0] - pp[1][0])
            w2 = math.fabs(pp[0][1] - pp[1][1])
            pcenter2 = (pp[0][0] + w1/2 ,pp[0][1] +w2/2 )
            #黑色矩形框的寬高比
            dp = w1 > w2 and (w2*1.0/w1) or (w1*1.0/w2)
            n = 0
#             cv2.rectangle(src,pp[0],pp[1],(0,0,255))
            if area > T6 and area <= T7 and dp >= T8 and dp <= T9:
#                 cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))
                for i in range(pp[0][1],pp[1][1]):
                    for j in range(pp[0][0],pp[1][0]):
                        if gray1[i][j] <= Tg2:
                            n += 1
                if n * 1.0 / (w1*w2) >= 0.80 and S * 1.0/area >= 0.03:
#                     cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))
                    d1 = math.fabs(pcenter1[0] - pcenter2[0])
                    d2 = math.fabs(pcenter1[1] - pcenter2[1])
                    if (d1 <= w1/2 and d2 <= 5) or (d2 <= w2/2 and d1 <= 5):
#                         cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))
                        cv2.rectangle(src,pp[0],pp[1],(0,0,255))   
#                     cv2.rectangle(src,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,0,255))

#                 print outPath+file[1]

def solve(img):
    #亮度阈值
    T1 = 182
    T2 = 0.6
    T3 = 1
    T4 = 2
    T5 = 250
    Tg1 = 50
    Tg2 = 100   
    T6 = 10
    T7 = 1000
    T8 = 0.2
    T9 = 0.6
    #调整图像大小
    img = cv2.resize(img,(500,375),img,0,0,cv2.INTER_AREA)

    #把图像转换为HSV
    hsv = img.copy()
    shape = img.shape
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            B,G,R = hsv[i][j]
            A = B > G and (B > R and B or R) or (G > R and G or R)

            if A < T1 or (B > 150 and G > 150 and R > 150) or (B < 100 and G < 100 and R < 100):
                hsv[i][j] = [0,0,0]
    
    #对图像进行灰度化
    gray = cv2.cvtColor(hsv,cv2.COLOR_RGB2GRAY) 
    gray1 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    imageProc(gray,gray1,img,T1,T2,T3,T4,T5,T6,T7,T8,T9,Tg1,Tg2)
    return img


def rgb_select(img,thresh=(200,255)):
    RG = img[0,:,:]
    binary = np.zeros_like(RG)
    binary[(RG>thresh[0]) & (RG<=thresh[1])] = 1
    return binary

def color_mask(hsv,low,high):
    mask = cv2.inRange(hsv,low,high)
    return mask

def apply_color_mask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    red_hsv_low = np.array([100,116,180])
    red_hsv_high = np.array([180,255,255])
    green_hsv_low = np.array([30,100,120])
    green_hsv_high = np.array([60,255,255])
    mask_red = color_mask(hsv,red_hsv_low,red_hsv_high)
    mask_green = color_mask(hsv,green_hsv_low,green_hsv_high)
    return mask_red

def combine_filters(img):
    #rg_binary = rgb_select(img,thresh=(100,150))
    #rg_binary = cv2.cvtColor(rg_binary,cv2.COLOR_RGB2GRAY)
    w_binary = apply_color_mask(img)
    w_binary[(w_binary!=0)] = 1
    combined_lsx = np.zeros_like(w_binary)
    #combined_lsx[((l_binary == 1) & (s_binary == 1) | (gradx == 1) | (yw_binary == 1))] = 1
    combined_lsx[((w_binary == 1))] = 1
    return w_binary


def solve2(img):
    ret = 0

    NROI=img[(int)(img.shape[0]/2):,:].copy()  #非ROI

    #将非ROI区域置0
    img[(int)(img.shape[0]/2):,:] = 0
    
    #转换为hsv空间
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #提取红色区域
    mask = apply_color_mask(img)

    #模糊
    blurred=cv2.blur(mask,(6,6))
    #二值化
    ret,binary=cv2.threshold(blurred,70,255,cv2.THRESH_BINARY)

    #使区域闭合无空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #腐蚀和膨胀
    '''
    腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    '''
    erode=cv2.erode(closed,None,iterations=3)
    dilate=cv2.dilate(erode,None,iterations=12)

    # 查找轮廓
    image,contours, hierarchy=cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    i=0
    res=img.copy()
    for con in contours:
        ret = 1

        #轮廓转换为矩形
        rect=cv2.minAreaRect(con)
        #矩形转换为box
        box=np.int0(cv2.boxPoints(rect))
        #在原图画出目标区域
        cv2.drawContours(res,[box],-1,(0,0,255),2)
        cv2.putText(res, 'red light', ([box][0][1][0]-25, [box][0][2][1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        #计算矩形的行列
        h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
        l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
        l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])

        #加上防错处理，确保裁剪区域无异常
        if h1-h2>0 and l1-l2>0:
            #裁剪矩形区域
            temp=img[h2:h1,l2:l1]
            i=i+1
            
    #恢复NROI部分
    res[(int)(img.shape[0]/2):,:] = NROI
    return res,ret


'''
#加载原图
img=cv2.imread('4.jpg')
NROI=img[(int)(img.shape[0]/2):,:].copy()  #非ROI

#将非ROI区域置0
img[(int)(img.shape[0]/2):,:] = 0
cv2.imshow('img',img)
cv2.imwrite('roi.jpg',img)
cv2.imshow('nroi',NROI)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
cv2.imwrite('hsv.jpg',hsv)

#提取红色区域
mask = apply_color_mask(img)
cv2.imshow('mask',mask)
cv2.imwrite('mask.jpg',mask)

#模糊
blurred=cv2.blur(mask,(6,6))
cv2.imshow('blurred',blurred)
cv2.imwrite('blurred.jpg',blurred)
#二值化
ret,binary=cv2.threshold(blurred,70,255,cv2.THRESH_BINARY)
cv2.imshow('blurred binary',binary)

#使区域闭合无空隙
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imshow('closed',closed)
cv2.imwrite('closed.jpg',closed)

#腐蚀和膨胀
erode=cv2.erode(closed,None,iterations=3)
cv2.imshow('erode',erode)
cv2.imwrite('erode.jpg',erode)
dilate=cv2.dilate(erode,None,iterations=12)
cv2.imshow('dilate',dilate)
cv2.imwrite('dilate.jpg',dilate)

# 查找轮廓
image,contours, hierarchy=cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print('轮廓个数：',len(contours))
i=0
res=img.copy()
for con in contours:
    #轮廓转换为矩形
    rect=cv2.minAreaRect(con)
    #矩形转换为box
    box=np.int0(cv2.boxPoints(rect))
    #在原图画出目标区域
    cv2.drawContours(res,[box],-1,(0,0,255),2)
    print([box])
    #计算矩形的行列
    h1=max([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
    h2=min([box][0][0][1],[box][0][1][1],[box][0][2][1],[box][0][3][1])
    l1=max([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
    l2=min([box][0][0][0],[box][0][1][0],[box][0][2][0],[box][0][3][0])
    cv2.putText(res, 'red light', ([box][0][1][0]-25, [box][0][2][1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    print('h1',h1)
    print('h2',h2)
    print('l1',l1)
    print('l2',l2)
    #加上防错处理，确保裁剪区域无异常
    if h1-h2>0 and l1-l2>0:
        #裁剪矩形区域
        temp=img[h2:h1,l2:l1]
        i=i+1
        #显示裁剪后的标志
        cv2.imshow('sign'+str(i),temp)
#显示画了标志的原图
res[(int)(img.shape[0]/2):,:] = NROI
cv2.imshow('res',res)
cv2.imwrite('res.jpg',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''