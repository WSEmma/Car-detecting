import cv2
import dlib
import time
import threading
import math

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('project_video.mp4')
fps = video.get(cv2.CAP_PROP_FPS)


WIDTH = 1280
HEIGHT = 720

# 摄像机的安装高度1米
h = 1000
# 视野范围最近点到摄像机的距离2m
dis = 2000
# 摄像机内部参数v0,fy
v0 = 2.61522808e+02
fy = 2.28026641e+03
# 1920*1080 分辨率 则V=1080
V = 720


# vv是传来的图像目标点的坐标
def distance_check(vv):
    aaa = math.atan(dis/h)
    bbb = math.atan(V/fy)
    ccc = math.atan((vv-v0)/fy)
    temp = math.tan(aaa + bbb + ccc)
    d = h/temp
    return d

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	#fps = 18
	speed = d_meters * fps * 3.6
	return speed

def solve(image,frameCounter,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete):
	
	ret = 0

	rectangleColor = (0, 255, 0)
	
	#speed = [None] * 1000

	resultImage = image.copy()

	for carID in carTracker.keys():
		trackingQuality = carTracker[carID].update(image)
			
		if trackingQuality < 10:
			carIDtoDelete.append(carID)
				
	for carID in carIDtoDelete:
		carTracker.pop(carID, None)
		carLocation1.pop(carID, None)
		carLocation2.pop(carID, None)
		
		
	if not (frameCounter % fps):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			
		for (_x, _y, _w, _h) in cars:
			x = int(_x)
			y = int(_y)
			w = int(_w)
			h = int(_h)
			
			x_bar = x + 0.5 * w
			y_bar = y + 0.5 * h
				
			matchCarID = None
			
			for carID in carTracker.keys():
				trackedPosition = carTracker[carID].get_position()
					
				t_x = int(trackedPosition.left())
				t_y = int(trackedPosition.top())
				t_w = int(trackedPosition.width())
				t_h = int(trackedPosition.height())
					
				t_x_bar = t_x + 0.5 * t_w
				t_y_bar = t_y + 0.5 * t_h
				
				if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
					matchCarID = carID
				
			if matchCarID is None:
				#print ('Creating new tracker ' + str(currentCarID))
					
				tracker = dlib.correlation_tracker()
				tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
				carTracker[currentCarID] = tracker
				carLocation1[currentCarID] = [x, y, w, h]

				currentCarID = currentCarID + 1
		
		#cv2.line(resultImage,(0,480),(1280,480),(255,0,0),5)

	flag = 0
	for carID1 in carTracker.keys():
		for carID2 in carTracker.keys():
			if carID1 == carID2:
				continue
			else:
				trackedPosition1 = carTracker[carID1].get_position()
				trackedPosition2 = carTracker[carID2].get_position()
				t_x1 = int(trackedPosition1.left())
				t_x2 = int(trackedPosition2.left())
				if abs(t_x1-t_x2) < 100:
					carTracker.pop(carID2, None)
					carLocation1.pop(carID2, None)
					carLocation2.pop(carID2, None)
					flag = 1
				break
		if flag == 1:
			break


	for carID in carTracker.keys():
		trackedPosition = carTracker[carID].get_position()
					
		t_x = int(trackedPosition.left())
		t_y = int(trackedPosition.top())
		t_w = int(trackedPosition.width())
		t_h = int(trackedPosition.height())
			
		cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
		cv2.putText(resultImage, 'car', (t_x, t_y-5),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
		d = distance_check(t_y+t_h)
		if d<49:
			cv2.putText(resultImage, 'dis'+str(int(d)), (t_x, t_y+t_h),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
			ret = 1
			
		carLocation2[carID] = [t_x, t_y, t_w, t_h]

	'''
	for i in carLocation1.keys():	
		if frameCounter % 1 == 0:
			[x1, y1, w1, h1] = carLocation1[i]
			[x2, y2, w2, h2] = carLocation2[i]
		
			carLocation1[i] = [x2, y2, w2, h2]
		
			if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
				if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
					speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

				#if y1 > 275 and y1 < 285:
				if speed[i] != None and y1 >= 180:
					cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
	'''

	return resultImage,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete,ret
