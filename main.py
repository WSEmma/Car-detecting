# coding:utf-8

from PyQt5 import QtCore,QtGui,QtWidgets,QtMultimedia,QtMultimediaWidgets
import sys
import qtawesome
import cv2
import threading
import image_process
import check_rglight
import speed_check
import numpy as np
import tkinter.messagebox
import pyttsx3

class MainUi(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.engine = pyttsx3.init()
        self.volume = self.engine.getProperty('volume')
        self.engine.setProperty('volume', self.volume+5)
        self.rate = self.engine.getProperty('rate')
        self.engine.setProperty('rate', self.rate+50)
        self.engine.setProperty("voice","HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
        
        self.setFixedSize(1560,720)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局

        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout) # 设置左侧部件布局为网格

        self.right_widget = QtWidgets.QLabel() # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout) # 设置右侧部件布局为网格

        self.main_layout.addWidget(self.left_widget,0,0,12,2) # 左侧部件在第0行第0列，占8行3列
        self.main_layout.addWidget(self.right_widget,0,2,12,10) # 右侧部件在第0行第3列，占8行9列
        self.setCentralWidget(self.main_widget) # 设置窗口主部件

        self.left_close = QtWidgets.QPushButton("") # 关闭按钮
        self.left_close.clicked.connect(self.close)
        self.left_visit = QtWidgets.QPushButton("") # 空白按钮
        self.left_mini = QtWidgets.QPushButton("")  # 最小化按钮

        self.left_close.setFixedSize(15, 15) # 设置关闭按钮的大小
        self.left_visit.setFixedSize(15, 15)  # 设置按钮大小
        self.left_mini.setFixedSize(15, 15) # 设置最小化按钮大小

        self.left_close.setStyleSheet('''QPushButton{background:#F76677;border-radius:5px;}QPushButton:hover{background:red;}''')
        self.left_visit.setStyleSheet('''QPushButton{background:#F7D674;border-radius:5px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet('''QPushButton{background:#6DDF6D;border-radius:5px;}QPushButton:hover{background:green;}''')

        #label设置
        self.left_label_1 = QtWidgets.QPushButton("数据源")
        self.left_label_1.setObjectName('left_label')
        self.left_label_2 = QtWidgets.QPushButton("文件操作")
        self.left_label_2.setObjectName('left_label')
        self.left_label_3 = QtWidgets.QPushButton("检测设置")
        self.left_label_3.setObjectName('left_label')

        #数据源相关按钮设置
        self.left_button_camera = QtWidgets.QRadioButton("摄像头")
        self.left_button_camera.setObjectName('left_button')
        self.left_button_camera.setChecked(True)    #默认视频源为摄像头
        self.left_button_camera.clicked.connect(self.set_camera)
        self.src_info = 1

        self.left_button_video = QtWidgets.QRadioButton("本地视频")
        self.left_button_video.setObjectName('left_button')
        self.left_button_video.clicked.connect(self.set_video)

        self.left_button_image = QtWidgets.QRadioButton("本地图片")
        self.left_button_image.setObjectName('left_button')
        self.left_button_image.clicked.connect(self.set_image)

        #操作相关按钮设置
        self.left_button_start = QtWidgets.QPushButton(qtawesome.icon('fa.play-circle',color='white'),"开始")
        self.left_button_start.setObjectName('left_button')
        self.left_button_start.clicked.connect(self.start)

        self.left_button_stop = QtWidgets.QPushButton(qtawesome.icon('fa.stop-circle',color='white'),"终止")
        self.left_button_stop.setObjectName('left_button')
        self.left_button_stop.clicked.connect(self.close_file)

        #检测相关按钮设置
        self.left_cb_pipeline = QtWidgets.QCheckBox("检测车道线")
        self.left_cb_pipeline.setObjectName('left_button')
        self.ispipeline = False
        self.left_cb_pipeline.stateChanged.connect(self.set_pipeline)
        self.left_cb_pipeline.setChecked(True)

        self.left_cb_car = QtWidgets.QCheckBox("检测车辆")
        self.left_cb_car.setObjectName('left_button')
        self.iscar = False
        self.left_cb_car.stateChanged.connect(self.set_car_check)

        self.left_cb_rglight = QtWidgets.QCheckBox("检测红绿灯")
        self.left_cb_rglight.setObjectName('left_button')
        self.isrglight = False
        self.left_cb_rglight.stateChanged.connect(self.set_rglight_check)

        #创建一个关闭事件
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.left_button_stop.setEnabled(False)

        #左侧部件布局
        self.left_layout.addWidget(self.left_mini, 0, 0,1,1)
        self.left_layout.addWidget(self.left_close, 0, 2,1,1)
        self.left_layout.addWidget(self.left_visit, 0, 1, 1, 1)
        self.left_layout.addWidget(self.left_label_1,1,0,1,3)
        self.left_layout.addWidget(self.left_button_camera, 2, 0,1,3)
        self.left_layout.addWidget(self.left_button_video, 3, 0,1,3)
        self.left_layout.addWidget(self.left_button_image, 4, 0,1,3)
        self.left_layout.addWidget(self.left_label_2, 5, 0,1,3)
        self.left_layout.addWidget(self.left_button_start, 6, 0,1,3)
        self.left_layout.addWidget(self.left_button_stop, 7, 0,1,3)
        self.left_layout.addWidget(self.left_label_3, 8, 0,1,3)
        self.left_layout.addWidget(self.left_cb_pipeline, 9, 0,1,3)
        self.left_layout.addWidget(self.left_cb_car, 10, 0,1,3)
        self.left_layout.addWidget(self.left_cb_rglight, 11, 0,1,3)

        #左侧部件美化
        self.left_widget.setStyleSheet('''
        QPushButton{border:none;color:white;}
        QPushButton#left_label{
            border:none;
            border-bottom:1px solid white;
            font-size:18px;
            font-weight:700;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        QPushButton#left_button:hover{border-left:4px solid red;font-weight:700;}
        QRadioButton{border:none;color:white;}
        QCheckBox{border:none;color:white;}
        QWidget#left_widget{
            background:gray;
            border-top:1px solid white;
            border-bottom:1px solid white;
            border-left:1px solid white;
            border-top-left-radius:10px;
            border-bottom-left-radius:10px;
        }
        ''')


        self.right_widget.setStyleSheet('''
        QWidget#right_widget{
            color:#232C51;
            background:white;
            border-top:1px solid darkGray;
            border-bottom:1px solid darkGray;
            border-right:1px solid darkGray;
            border-top-right-radius:10px;
            border-bottom-right-radius:10px;
        }
        QLabel#right_lable{
            border:none;
            font-size:16px;
            font-weight:700;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        }
        ''')
        

        #self.setWindowOpacity(0.9) # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # 设置窗口背景透明
        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint) # 隐藏边框
        self.main_layout.setSpacing(0)

    def set_camera(self):
        self.src_info = 1

    def set_video(self):
        self.src_info = 2

    def set_image(self):
        self.src_info = 3

    def start(self):
        if self.src_info == 2:
            self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self.right_widget, 'Choose file', '', '*.mp4')
            self.cap = cv2.VideoCapture(self.fileName)
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        elif self.src_info == 1:
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        elif self.src_info == 3:
            self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self.right_widget, 'Choose file', '', '*.jpg')
            self.img = cv2.imread(self.fileName)
        th = threading.Thread(target=self.Display)
        th.start()

    def close_file(self):
        self.stopEvent.set()
    
    def Display(self):
        self.left_button_start.setEnabled(False)
        self.left_button_stop.setEnabled(True)
        if self.src_info == 3:
            if self.img is None:
                self.stopEvent.set()
                self.stopEvent.clear()
                self.right_widget.clear()
                self.left_button_stop.setEnabled(False)
                self.left_button_start.setEnabled(True)
                return
            self.img = cv2.resize(self.img,(1280, 720),self.img,0,0,cv2.INTER_AREA)
            if self.isrglight == True:
                self.img,ret = check_rglight.solve2(self.img)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
            if self.ispipeline == True:
                M,Minv = image_process.getM()
                self.img,left_fit,right_fit,ret = image_process.solve(self.img,M,Minv)
            if self.iscar == True:
                frameCount = 0
                carTracker = {}
                carNumbers = {}
                carLocation1 = {}
                carLocation2 = {}
                currentCarID = 0
                carIDtoDelete = []
                self.img,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete,ret = speed_check.solve(self.img,0,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete)
            image = QtGui.QImage(self.img.data, self.img.shape[1], self.img.shape[0], QtGui.QImage.Format_RGB888)
            self.right_widget.setPixmap(QtGui.QPixmap.fromImage(image))
            self.right_widget.setScaledContents(True)
            while True:
                if True == self.stopEvent.is_set():
                    # 关闭事件置为未触发，清空显示label
                    self.stopEvent.clear()
                    self.right_widget.clear()
                    self.left_button_stop.setEnabled(False)
                    self.left_button_start.setEnabled(True)
                    break
        else:
            lhas_played = 0
            has_played = 0
            chas_played = 0
            frameCount = 0
            carTracker = {}
            carNumbers = {}
            carLocation1 = {}
            carLocation2 = {}
            currentCarID = 0
            carIDtoDelete = []
            left_fit = []
            right_fit = []
            first = 1
            lret = 0
            cret = 0
            ret = 0
            M,Minv = image_process.getM()
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if success == False:
                    break
                frame = cv2.resize(frame,(1280, 720),frame,0,0,cv2.INTER_AREA)
                if self.isrglight == True:
                    frame,lret = check_rglight.solve2(frame)
                # RGB转BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if self.ispipeline == True:
                    if first%30 == 1:
                        frame,left_fit,right_fit,ret = image_process.solve(frame,M,Minv)
                    elif left_fit is None or right_fit is None:
                        frame,left_fit,right_fit,ret = image_process.solve(frame,M,Minv)
                    else:
                        frame,left_fit,right_fit,ret = image_process.solve_by_previous(frame,left_fit,right_fit,M,Minv)
                    first = first + 1
                if self.iscar == True:
                    frame,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete,cret = speed_check.solve(frame,frameCount+1,carTracker,carNumbers,carLocation1,carLocation2,currentCarID,carIDtoDelete)
                    frameCount = frameCount + 1
                if lret == 1 and lhas_played == 0:
                    lhas_played = 1
                    th = threading.Thread(target=self.light_alarm)
                    th.start()
                elif cret == 1 and chas_played == 0:
                    th = threading.Thread(target=self.car_alarm)
                    th.start()
                    chas_played = 1
                elif ret == -1 and has_played == 0:
                    th = threading.Thread(target=self.cross_alarm)
                    th.start()
                    has_played = has_played + 1
                elif ret == 1 and has_played == 0:
                    th = threading.Thread(target=self.mcross_alarm)
                    th.start()
                    has_played = has_played + 1
                if ret == 0 and has_played > 30:
                    has_played = 0
                if has_played > 0:
                    has_played = has_played + 1
                if cret == 0:
                    chas_played = 0
                if lret == 0:
                    lhas_played = 0
                image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
                self.right_widget.setPixmap(QtGui.QPixmap.fromImage(image))
                self.right_widget.setScaledContents(True)

                if self.src_info == 1:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(int(1000 / self.frameRate))

                # 判断关闭事件是否已触发
                if True == self.stopEvent.is_set():
                    # 关闭事件置为未触发，清空显示label
                    self.stopEvent.clear()
                    self.right_widget.clear()
                    self.left_button_stop.setEnabled(False)
                    self.left_button_start.setEnabled(True)
                    break
            if not self.cap.isOpened() and self.src_info == 1:
                print("请连接usb摄像头!")
                tkinter.messagebox.showwarning("Warning","请连接USB摄像头！")

        if False == self.stopEvent.is_set():
            self.stopEvent.set()
            self.stopEvent.clear()
            self.right_widget.clear()
            self.left_button_stop.setEnabled(False)
            self.left_button_start.setEnabled(True)
        
    def light_alarm(self):
        try:
            self.engine.say("前方红灯,请注意！")
            self.engine.runAndWait()
        except RuntimeError as rte:
            print('RuntimeError',rte)

    def cross_alarm(self):
        try:
            self.engine.say("正在越道行驶,请注意！")
            self.engine.runAndWait()
        except RuntimeError as rte:
            print('RuntimeError',rte)

    def mcross_alarm(self):
        try:
            self.engine.say("即将越道行驶,请注意！")
            self.engine.runAndWait()
        except RuntimeError as rte:
            print('RuntimeError',rte)

    def car_alarm(self):
        try:
            self.engine.say("有较近车辆,请注意！")
            self.engine.runAndWait()
        except RuntimeError as rte:
            print('RuntimeError',rte)

    def set_pipeline(self):
        if self.ispipeline == False:
            self.ispipeline = True
        else:
            self.ispipeline = False
    
    def set_rglight_check(self):
        if self.isrglight == False:
            self.isrglight = True
        else:
            self.isrglight = False

    def set_car_check(self):
        if self.iscar == False:
            self.iscar = True
        else:
            self.iscar = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = MainUi()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()