import cv2
import numpy as np
import math
import os
import glob

import time
import sys
from threading import Thread
import csv

import timeit
import math


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)    

threading_Time = 5/1000.



class hashDetector():
    def __init__(self):
        pass

    def image_to_hash(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (32, 32))
        avg = gray.mean()
        bins = 1 * (gray > avg)
        return bins

    def hamming_distance(self, a, b):
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        distance = (a != b).sum()
        return distance

    def getContours(img): #이미지 윤곽선 그리기
        biggest = np.array([])
        pts = np.array([])
        maxArea = 0
        contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            #print(area)
            if area>7000:
                cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3) #윤곽선 그리기
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,0.02*peri,True)
                x,y,w,h=cv2.boundingRect(approx)
                pts=np.float32([[x-30,y-30],[x+w+30,y-30],[x-30,y+h+30],[x+w+30,y+h+30]])
                cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,255,0),5)
                
                if area >maxArea and len(approx) >1:
                    biggest = approx
                    maxArea = area
        #cv2.drawContours(imgCopy, biggest, -1, (0, 255, 0), 10) #점 그리기
        return biggest,pts




    def getWarp(self,img,pts):
        pts1 = pts
        pts2 = np.float32([[0, 0], [frameWidth, 0], [0, frameHeight], [frameWidth, frameHeight]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOutput = cv2.warpPerspective(img, matrix, (frameWidth, frameHeight))

        imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
        imgCropped = cv2.resize(imgCropped,(frameWidth,frameHeight))

        return imgCropped

    def get_image(self):
        _, image = cap.read()
        biggest,pts = self.getContours(image)
        imgWarped=self.getWarp(image,pts)
        return image




while True:
    _, image = cap.read()
    imgCopy = image.copy()

    start_t = timeit.default_timer()

    hashDetect=hashDetector()
    cur_path = os.path.dirname(__file__)    
    img_path = cur_path + '/alphabet/'
    img_path2 = cur_path + '/arrow/'
    imgBlank = np.zeros_like(image)

    biggest,pts = hashDetector.getContours(imgCopy)

    if biggest.size !=0:
        imgWarped=hashDetector.getWarp(imgCopy,pts)        
        # imageArray = ([img,imgThres],
        #           [imgContour,imgWarped])
        imageArray = ([imgCopy, imgWarped])
        cv2.imshow("ImageWarped", imgWarped)
    else:
        imgWarped=imgBlank
        # imageArray = ([img, imgThres],
        #               [img, img])
        imageArray = ([imgCopy, img])

    img_A = cv2.imread(img_path + 'A.jpg')
    img_B = cv2.imread(img_path + 'B.jpg')
    img_C = cv2.imread(img_path + 'C.jpg')
    img_D = cv2.imread(img_path + 'D.jpg')
    img_E = cv2.imread(img_path + 'E.jpg')
    img_W = cv2.imread(img_path + 'W.jpg')
    img_s = cv2.imread(img_path + 'S.jpg')
    img_N = cv2.imread(img_path + 'N.jpg')
    img_R = cv2.imread(img_path2 + 'right.jpg')
    img_L = cv2.imread(img_path2 + 'left.jpg')

    A_hash = hashDetect.image_to_hash(img_A)
    B_hash = hashDetect.image_to_hash(img_B)
    C_hash = hashDetect.image_to_hash(img_C)
    D_hash = hashDetect.image_to_hash(img_D)
    E_hash = hashDetect.image_to_hash(img_E)
    W_hash = hashDetect.image_to_hash(img_W)
    S_hash = hashDetect.image_to_hash(img_s)
    N_hash = hashDetect.image_to_hash(img_N)
    R_hash = hashDetect.image_to_hash(img_R)
    L_hash = hashDetect.image_to_hash(img_L)
    img = hashDetect.get_image()
    bins = hashDetect.image_to_hash(img)
    #print(bins)    
        
    img_hash = hashDetect.image_to_hash(img)
    A_dst = hashDetect.hamming_distance(A_hash, img_hash)
    B_dst = hashDetect.hamming_distance(B_hash, img_hash)
    C_dst = hashDetect.hamming_distance(C_hash, img_hash)
    D_dst = hashDetect.hamming_distance(D_hash, img_hash)
    E_dst = hashDetect.hamming_distance(E_hash, img_hash)
    W_dst = hashDetect.hamming_distance(W_hash, img_hash)
    S_dst = hashDetect.hamming_distance(S_hash, img_hash)
    N_dst = hashDetect.hamming_distance(N_hash, img_hash)
    R_dst = hashDetect.hamming_distance(R_hash, img_hash)
    L_dst = hashDetect.hamming_distance(L_hash, img_hash)   

    cv2.imshow('HashDectect', img)
    terminate_t = timeit.default_timer()
    FPS = int(1./(terminate_t - start_t ))  

    print(f"HashDetect FPS : {FPS}")  


    if A_dst/1024 < 0.15:
        print("A:{0}".format(A_dst/1024))
    elif B_dst/1024 < 0.10:
        print("B:{0}".format(B_dst/1024))
    elif C_dst/1024 < 0.09:
        print("C:{0}".format(C_dst/1024))
    elif D_dst/1024 < 0.13:
        print("D:{0}".format(D_dst/1024))
    elif E_dst/1024 < 0.10:
        print("E:{0}".format(E_dst/1024))
        print(E_dst/1024)
    elif W_dst/1024 < 0.10:
        print("W:{0}".format(W_dst/1024))
    elif S_dst/1024 < 0.10:
        print("S:{0}".format(S_dst/1024))
    elif N_dst/1024 < 0.10:
        print("N:{0}".format(N_dst/1024))

    elif R_dst/1024 < 0.10:
        print("Right:{0}".format(R_dst/1024))
    elif L_dst/1024 < 0.10:
        print("Left:{0}".format(L_dst/1024))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if __name__ == "__main__":    
    cv2.imshow('HashDectect', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()