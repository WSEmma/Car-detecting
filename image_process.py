import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

#解决cv2库色彩空间和matplotlib库色彩空间排布不一样的情况
def bgr2rgb(img):
    img_rgb = np.zeros(img.shape,img.dtype)
    img_rgb[:,:,0] = img[:,:,2]
    img_rgb[:,:,1] = img[:,:,1]
    img_rgb[:,:,2] = img[:,:,0]
    return img_rgb


#获取透视变换的矩阵
def getM():
    '''
    leftupperpoint = [520, 410]
    rightupperpoint = [820, 410]
    leftlowerpoint = [390, 530]
    rightlowerpoint = [950, 530]

    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[100, 0], [100, 720], [1100, 0], [1100, 720]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    '''
    leftupperpoint = [568, 470]
    rightupperpoint = [717, 470]
    leftlowerpoint = [260, 680]
    rightlowerpoint = [1043, 680]

    src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
    dst = np.float32([[200, 0], [200, 680], [1000, 0], [1000, 680]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv

#透视变换
def perspective_transformation(img,M):
    #src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])   #源点
    #dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])        #目标点
    img_size = (img.shape[1],img.shape[0])
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped

#使用索贝尔算子进行边缘检测，由于车道线通常是垂直方向的线，故默认采用x方向的索贝尔算子
def abs_sobel_thresh(img,orient='x',thresh_min=0,thresh_max=255):
    #转换为灰度图片
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #使用cv2.Sobel()计算x方向或y方向的导数，图像为unit8类型（无符号整数），故要求取绝对值
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))
    #转换为unit8型数据并进行阈值过滤，返回对应二进制图（黑白图）
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary

#车道线通常为黄色或白色，提取hsv色彩空间的黄色和白色
def color_mask(hsv,low,high):
    mask = cv2.inRange(hsv,low,high)
    return mask

def apply_color_mask(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    yellow_hsv_low = np.array([19,34,46])
    yellow_hsv_high = np.array([34,255,255])
    white_hsv_low = np.array([0,0,185])
    white_hsv_high = np.array([180,30,255])
    mask_yellow = color_mask(hsv,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(hsv,white_hsv_low,white_hsv_high)
    mask = cv2.bitwise_or(mask_yellow,mask_white)
    return mask

'''
#使用hls颜色空间的L通道提取白色车道线，效果不是很好
def hls_select(img,channel='l',thresh=(0,255)):
    #转换为HLS通道图
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    if channel == 'h':
        x = hls[:,:,0]
    elif channel == 'l':
        x = hls[:,:,1]
    else:
        x = hls[:,:,2]
    binary = np.zeros_like(x)
    binary[(x>thresh[0]) & (x<=thresh[1])] = 1
    return binary
'''

def combine_filters(img):
    gradx = abs_sobel_thresh(img,orient='x',thresh_min=50,thresh_max=255)
    #l_binary = hls_select(img,channel='l',thresh=(100,200))
    #s_binary = hls_select(img,channel='s',thresh=(100,255))
    yw_binary = apply_color_mask(img)
    yw_binary[(yw_binary!=0)] = 1
    combined_lsx = np.zeros_like(gradx)
    #combined_lsx[((l_binary == 1) & (s_binary == 1) | (gradx == 1) | (yw_binary == 1))] = 1
    combined_lsx[((gradx == 1) | (yw_binary == 1))] = 1
    return combined_lsx

def find_line_fit(img,nwindows=9,margin=100,minpix=50):
    histogram = np.sum(img[img.shape[0]//2:,:],axis=0)
    #创建待绘制的输出图像来显示结果
    out_img = np.dstack((img,img,img))*255
    #将图片分为左右两个部分并获取其峰值
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #设定滑动窗口的高度
    height = np.int(img.shape[0]//nwindows)
    #找出所有非0值像素
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #当前位置被滑动窗口更新
    leftx_current = leftx_base
    rightx_current = rightx_base
    #创建空表来得到左右直线像素
    left_lane_inds = []
    right_lane_inds = []

    #逐个检查滑动窗口
    for window in range(nwindows):
        #找出窗口边界
        win_y_low = img.shape[0] - (window+1)*height
        win_y_high = img.shape[0] - window*height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        #绘制窗口以可视化
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
        #标识在窗口内的非零像素
        good_left_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
        #加入到list中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #比最小像素大，重定位下一个窗口
        if len(good_left_inds)>minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #合并数组
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #提取左右线像素位置
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #标志不同颜色，方便观察
    out_img[nonzeroy[left_lane_inds],nonzerox[left_lane_inds]] = [255,0,0]
    out_img[nonzeroy[right_lane_inds],nonzerox[right_lane_inds]] = [0,0,255]

    #构建拟合的二阶多项式
    if lefty.any() and leftx.any():
        left_fit = np.polyfit(lefty,leftx,2)
    else:
        left_fit = None
    if righty.any() and rightx.any():
        right_fit = np.polyfit(righty,rightx,2)
    else:
        right_fit = None
    
    return left_fit,right_fit,out_img

def get_fit_xy(img,left_fit,right_fit):
    ploty = np.linspace(0,img.shape[0]-1,img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx,right_fitx,ploty

def project_back(wrap_img,origin_img,left_fitx,right_fitx,ploty,M):
    warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right))

    #绘制线
    cv2.fillPoly(color_warp,np.int_([pts]),(0,0,255))

    #将图像恢复原视角
    newwarp = perspective_transformation(color_warp,M)
    #将结果与原图像结合
    result = cv2.addWeighted(origin_img,1,newwarp,0.3,0)
    return result

def calculate_curv_and_pos(img,left_fit, right_fit):
    #计算拟合的道路线坐标
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    '''
    #将x，y从像素对应转换成米
    ym_per_pix = 30/720 #y方向上一像素每米
    xm_per_pix = 3.7/700 #x方向上一像素每米
    y_eval = np.max(ploty)
    #构建在世界坐标下新的多项式
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    #对曲线计算其相应曲率
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curvature = ((left_curverad + right_curverad) / 2)
    #print(curvature)
    '''
    lane_width = np.absolute(leftx[719] - rightx[719])
    lane_xm_per_pix = 3.7 / lane_width
    veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
    cen_pos = ((img.shape[1] * lane_xm_per_pix) / 2.)
    distance_from_center = cen_pos - veh_pos
    return distance_from_center

def find_line_by_previous(binary_warped,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if lefty.any() and leftx.any():
        left_fit = np.polyfit(lefty,leftx,2)
    else:
        left_fit = None
    if righty.any() and rightx.any():
        right_fit = np.polyfit(righty,rightx,2)
    else:
        right_fit = None

    return left_fit, right_fit

#预警
def alarm(pos_from_center):
    ret = 0
    half_car_width = 2.1/2
    half_line_width = 3.7/2
    pos_from_center = abs(pos_from_center)
    if (pos_from_center+half_car_width)>half_line_width:
        ret = -1
    elif (half_line_width-pos_from_center-half_car_width) <= 0.5:
        ret = 1
    return ret

def draw_values(img,distance_from_center,ret):
    font = cv2.FONT_HERSHEY_SIMPLEX
    #radius_text = "Radius of Curvature: %sm"%(round(curvature))
    
    if distance_from_center>0:
        pos_flag = 'right'
    else:
        pos_flag= 'left'
        
    #cv2.putText(img,radius_text,(100,100), font, 1,(255,255,255),2)
    center_text = "Vehicle is %.3fm %s of center"%(abs(distance_from_center),pos_flag)
    cv2.putText(img,center_text,(100,100), font, 1,(255,255,255),2)
    if ret == 1:
        alarm_text = "Be Careful!You might cross the line!"
        alarm_color = (255,215,0)
    elif ret == -1:
        alarm_text = "You are crossing the line!"
        alarm_color = (255,0,0)
    else:
        alarm_text = "You are driving normally"
        alarm_color = (0,255,0)
    cv2.putText(img,alarm_text,(100,150), font, 1,alarm_color,2)
    return img

#焦距标定
def getf(m,n,theta):
    f = n*math.cos(theta)/(m-math.sin(theta))
    return f

#计算汽车偏离车道线的角度与距离
def calculate(h,theta,f,m,n):
    anger = 1/math.tan(n*math.cos(theta)+f*math.sin(theta)/m/f)
    distance = h*math.cos(anger)-m*h*math.sin(theta)*math.sin(anger)/m/math.cos(theta)
    return anger,distance

def solve(img,M,Minv):
    wrap_img = perspective_transformation(img,M)
    binary = combine_filters(wrap_img)
    left_fit, right_fit, out_img = find_line_fit(binary)
    if left_fit is None or right_fit is None:
        return img,left_fit,right_fit,0
    else:
        left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)
        pos_from_center = calculate_curv_and_pos(binary,left_fit, right_fit)
        ret = alarm(pos_from_center)
        img = project_back(binary, img, left_fitx, right_fitx, ploty, Minv)
        img = draw_values(img,pos_from_center,ret)
    return img,left_fit,right_fit,ret

def solve_by_previous(img,left_fit,right_fit,M,Minv):
    wrap_img = perspective_transformation(img,M)
    binary = combine_filters(wrap_img)
    left_fit, right_fit = find_line_by_previous(binary,left_fit,right_fit)
    if left_fit is None or right_fit is None:
        return img,left_fit,right_fit,0
    else:
        left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)
        pos_from_center = calculate_curv_and_pos(binary,left_fit, right_fit)
        ret = alarm(pos_from_center)
        img = project_back(binary, img, left_fitx, right_fitx, ploty, Minv)
        img = draw_values(img,pos_from_center,ret)
    return img,left_fit,right_fit,ret

#M,Minv = getM()

'''
leftupperpoint = [568, 470]
rightupperpoint = [717, 470]
leftlowerpoint = [260, 680]
rightlowerpoint = [1043, 680]

src = np.float32([leftupperpoint, leftlowerpoint, rightupperpoint, rightlowerpoint])
dst = np.float32([[200, 0], [200, 680], [1000, 0], [1000, 680]])
M = cv2.getPerspectiveTransform(src,dst)
Minv = cv2.getPerspectiveTransform(dst,src)
img = cv2.imread("test.jpg")
img = bgr2rgb(img)
'''

'''
wrap_img = perspective_transformation(img,M)
f, axs = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
axs[0].imshow(img)
axs[0].set_title('Original Image', fontsize=18)
axs[1].imshow(wrap_img, cmap='gray')
axs[1].set_title('Transform', fontsize=18)
plt.show()
'''


'''
wrap_img = perspective_transformation(img,M)
gradx = abs_sobel_thresh(wrap_img,orient='x',thresh_min=50,thresh_max=255)
f, axs = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
axs[0].imshow(wrap_img)
axs[0].set_title('Original Image', fontsize=18)
axs[1].imshow(gradx, cmap='gray')
axs[1].set_title('Sobel_x_filter', fontsize=18)
plt.show()
cv2.imwrite('sobel.jpg',gradx)
'''


'''
wrap_img = perspective_transformation(img,M)
r_binary = rgb_select(wrap_img, thresh=(165, 255))
yw_binary = apply_color_mask(wrap_img)
f, axs = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
axs[0].imshow(wrap_img)
axs[0].set_title('R filter', fontsize=18)
axs[1].imshow(yw_binary, cmap='gray')
axs[1].set_title('Yellow white filter', fontsize=18)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
'''

'''
wrap_img = perspective_transformation(img,M)
l_binary = hls_select(wrap_img, channel='l', thresh=(90, 200))
s_binary = hls_select(wrap_img, channel='s', thresh=(90, 255))
h_binary = hls_select(wrap_img, channel='h', thresh=(90, 255))
f, axs = plt.subplots(2, 2, figsize=(16, 9))
f.tight_layout()
axs[0, 0].imshow(wrap_img)
axs[0, 0].set_title('Original Image', fontsize=18)
axs[0, 1].imshow(h_binary, cmap='gray')
axs[0, 1].set_title('H channal filter', fontsize=18)
axs[1, 0].imshow(s_binary, cmap='gray')
axs[1, 0].set_title('S channal filter', fontsize=18)
axs[1, 1].imshow(l_binary, cmap='gray')
axs[1, 1].set_title('L channal filter', fontsize=18)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
'''

'''
wrap_img = perspective_transformation(img,M)
binary = combine_filters(wrap_img)
f, axs = plt.subplots(1, 2, figsize=(16, 9))
f.tight_layout()
axs[0].imshow(wrap_img)
axs[0].set_title('Original', fontsize=18)
axs[1].imshow(binary, cmap='gray')
axs[1].set_title('combine filters', fontsize=18)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
'''

'''
wrap_img = perspective_transformation(img,M)
binary = combine_filters(wrap_img)
left_fit, right_fit, out_img = find_line_fit(binary)
left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)

fig = plt.figure(figsize=(16, 9))
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='white', linewidth=3.0)
plt.plot(right_fitx, ploty, color='white',  linewidth=3.0)
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
'''

'''
wrap_img = perspective_transformation(img,M)
binary = combine_filters(wrap_img)
left_fit, right_fit, out_img = find_line_fit(binary)
left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)
result = project_back(binary, img, left_fitx, right_fitx, ploty, Minv)
fig = plt.figure(figsize=(16, 9))
plt.imshow(result)
plt.show()
'''

'''
video = cv2.VideoCapture("challenge_video.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#M,Minv = getM()
success, frame = video.read()  
index = 1
while success:
    cv2.imwrite('test1.jpg',frame)
    frame = bgr2rgb(frame)
    wrap_img = perspective_transformation(frame,M)
    binary = combine_filters(wrap_img)
    left_fit, right_fit, out_img = find_line_fit(binary)
    left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)
    pos_from_center = calculate_curv_and_pos(binary,left_fit, right_fit)
    ret = alarm(pos_from_center)
    frame = project_back(binary, frame, left_fitx, right_fitx, ploty, Minv)
    frame = bgr2rgb(frame)
    frame = draw_values(frame,pos_from_center,ret)
    #area_img = utils.draw_area(undist,thresholded_wraped,Minv,left_fit, right_fit)
    #frame = cv2.putText(frame, str(index), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imshow("new video", frame)
    cv2.waitKey(int(1000 / int(fps)))
    success, frame = video.read()
    index += 1

video.release()
'''