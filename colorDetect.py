from email.utils import collapse_rfc2231_value
import cv2
import math
import numpy as np


########## 대회에서는 mini_cts5_py3.py로 hsv 읽기#################
myColors = [[0, 139, 59, 140, 174, 134],    #blue hmin smin vmin  hmax smax vmax
            [110, 110, 170, 130, 130, 190], #red
            [20, 120, 110, 40, 140, 130]]   #black
myColorValues = [[0, 0, 255],
                [255, 0, 0],
                [0, 255, 0]]
myText = ["X", "Citizen", "Danger zone"]

class colorDetector():
    def __init__(self):
        pass

    def findColor(self,image, myColors):
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)    # BGR을 HSV로 변환(YUV 이용)
        count = 0
        newPoints = []
        for color in myColors:
            lower = np.array(color[0:3])
            upper = np.array(color[3:6])
            mask = cv2.inRange(imgHSV, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("Result11", result)
            a,b= self.getContours(mask)
           
            #cv2.circle(imgResult, (x, y), 15, myColorValues[count], cv2.FILLED)  # 수정
            cv2.putText(imgResult, text=myText[count], org=(a, b), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
            if a != 0 and b != 0:
                newPoints.append([a, b, count])
            count += 1
            #cv2.imshow(str(color[0]),mask)
        return newPoints

    def getContours(self,img):
        contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x, y, w, h = 0, 0, 0, 0
        for cnt in contours:
           area = cv2.contourArea(cnt)
           if area > 400:
               #cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgResult,(x,y),(x+w,y+h),(0,255,0),5)
        return x+w//2, y 



if __name__ == "__main__":
    frameWidth = 640    
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150) 
    while True:
       success, img = cap.read()
       imgResult = img.copy()
       imgBlank = np.zeros_like(img)
       colorDetect=colorDetector()
       colorDetect.findColor(img, myColors)
       #imgStack = stackImages(0.8,([imgResult,imgBlank,imgBlank],[imgBlank,imgBlank,imgBlank]))
       cv2.imshow("Result", imgResult)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
 

 