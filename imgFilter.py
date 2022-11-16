import cv2
import numpy as np


###################이미지 기본 셋팅########################

def hsv(image):  # 노락색만 추출
    imgHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([19, 48, 130])
    upper = np.array([70, 255, 255])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

def preProcessing(image):  # 이미지 엣지 추출하기(이미지 초기작업)
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

################################################################################