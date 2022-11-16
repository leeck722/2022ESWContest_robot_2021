import cv2
import math
import numpy as np
import imgFilter
import stackImages
import timeit
import lineDetect as ld

class lineDetector():

    def __init__(self):
        pass
    
    def makePoints(self,image, line):
       height, width = image.shape  # 이미지의 크기 불러오기
       slope, intercept = line # line 값 slop, intercept에 넣기
       y1 = height  # 프레임 높이
       y2 = 0  # 프레임으로부터 0 지점
    
       # 화면 내로 x1, x2의 영역 설정
       x1 = int((y1 - intercept) / slope)
       x2 = int((y2 - intercept) / slope)
       return [[x1, y1, x2, y2]]   

    def getAverageSlope(self,image, lines):
    
        height, width = image.shape # 이미지의 크기 불러오기

        left = [] # 필터링된 왼쪽 차선 좌표 값 배열로 추출
        right = [] # 필터링된 오른쪽 차선 좌표 값 배열로 추출
        
        if lines is None: # 선이 인식이 되지 않는 경우
            return left, right # 빈 배열 리턴

        else :
            left_fit = [] 
            right_fit = []

            for line_segment in lines:
                for x1, y1, x2, y2 in line_segment:
                    fit = np.polyfit((x1, x2), (y1, y2), 1) # 두 점 사이의 일차식 형성
                    slope = fit[0] # 기울기 구하기
                    intercept = fit[1] # y 절편 구하기
                    if slope < - 0.01 : # 기울기가 음수인 경우 /선으로 인식
                        left_fit.append((slope, intercept)) # left_fit 배열에 기울기와 절편 데이터 입력
                            
                    elif slope > 0.01 : # 기울기가 양수인 경우 \선으로 인식
                        right_fit.append((slope, intercept)) # right_fit 배열에 기울기와 절편 데이터 입력

                    else :
                        continue

            if len(left_fit) > 0 and len(right_fit) > 0: # left_fit와 right_fit에 입력 데이터가 있는 경우
                left_fit_average = np.average(left_fit, axis=0) # left_fit 값들의 평균
                right_fit_average = np.average(right_fit, axis=0) # left_fit 값들의 평균
                left.append(self.makePoints(image, left_fit_average)) # left 배열에 left_fit_average로 구성된 선 추가
                right.append(self.makePoints(image, right_fit_average)) # right 배열에 right_fit_average로 구성된 선 추가

            elif len(left_fit) > 0 and len(right_fit) <= 0: # left_fit에만 입력 데이터가 있는 경우 (/일때)
                left_fit_average = np.average(left_fit, axis=0) # left_fit 값들의 평균                
                left.append(self.makePoints(image, left_fit_average)) # left 배열에 left_fit_average로 구성된 선 추가
                
            elif len(left_fit) <= 0 and len(right_fit) > 0: # right_fit에만 입력 데이터가 있는 경우 (\일때)
                right_fit_average = np.average(right_fit, axis=0) # left_fit 값들의 평균
                right.append(self.makePoints(image, right_fit_average)) # right 배열에 right_fit_average로 구성된 선 추가

            return left, right
        
    def angleToHead(self, image, left, right):
        height, width, _ = image.shape
        head_image = np.zeros_like(image)
        y_offset = int(height * 2/3)
        angle_to_head_deg = 0
        line_start = 0, y_offset
        line_end = int(width), y_offset
        detection = [False, False] # detction[0] - \ 차선 인식 여부, detection[1] - / 차선 인식 여부

        cv2.line(image, line_start, line_end, (0,255,0), 2) # 조향 각 판단을 위한 기준 선 생성

        if left or right is not None :

            if len(left) == 1 and len(right) == 1 : # \, / 모두 인식
                detection = [True, True] # detection[0] - 왼쪽 차선 인식, detection[1] - 오른쪽 차선 인식
                l = left[0][0]
                r = right[0][0]
                left_slope=(l[2]-l[0])/(l[3]-l[1])
                right_slope=(r[2]-r[0])/(r[3]-r[1])

                x_left_center = int((y_offset-l[1])*left_slope)+int(l[0])
                x_right_center = int((y_offset-r[1])*right_slope)+int(r[0])
                if left_slope > - 0.1 :
                    cv2.line(image, (l[0], l[1]), (l[2], l[3]), [255, 0, 0], 6)
                    cv2.line(image, (r[0], r[1]), (r[2], r[3]), [255, 0, 255], 6)
                elif right_slope <  0.1 :
                    cv2.line(image, (l[0], l[1]), (l[2], l[3]), [255, 0, 255], 6)
                    cv2.line(image, (r[0], r[1]), (r[2], r[3]), [0, 0, 255], 6)

            elif len(left) == 0 and len(right) == 1: # \ 선만 인식
                detection = [False, True] # detection[0] - 왼쪽 차선 인식 실패, detection[1] - 오른쪽 차선 인식
                r = right[0][0]
                x_right_center = int((y_offset-r[1])*(r[2]-r[0])/(r[3]-r[1]))+int(r[0])

                cv2.putText(image,'LEFT',(20, 50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3) # LEFT 표기
                cv2.circle(image, (x_right_center, y_offset), 10, [255, 0, 0], -1) # 기준 선 상에 원으로 인식 위치 표시
                cv2.line(image, (x_right_center, y_offset), (x_right_center, 0), [0, 255, 0], 6) # 직진을 위한 기준 선 표시

                angle_to_head_deg = math.degrees(math.atan2(int(r[1]-y_offset),int(r[0]-x_right_center)))-90 # 각도 계산            
                cv2.putText(image,f'Degree : {int(angle_to_head_deg)}',(20, 90),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3) # 각도 표시
                if angle_to_head_deg<-30:
                    print('회전1')#음수
                #print(angle_to_head_deg)#음수

                cv2.line(image, (r[0], r[1]), (r[2], r[3]), [255, 0, 0], 6) # \ 인식된 선 표시 -> 파란색

            elif len(left) == 1 and len(right) == 0 : # / 선만 인식
                detection = [True, False] # detection[0] - 왼쪽 차선 인식, detection[1] - 오른쪽 차선 인식 실패
                l = left[0][0]
                x_left_center = int((y_offset-l[1])*(l[2]-l[0])/(l[3]-l[1]))+int(l[0])

                cv2.putText(image,'RIGHT',(20, 50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3) # RIGHT 표기
                cv2.circle(image, (x_left_center, y_offset), 10, [0, 0, 255], -1) # 기준 선 상에 원으로 인식 위치 표시
                cv2.line(image, (x_left_center, y_offset), (x_left_center, 0), [0, 255, 0], 6) # 직진을 위한 기준 선 표시

                angle_to_head_deg = math.degrees(math.atan2(int(l[1]-y_offset),int(l[0]-x_left_center)))-90 # 각도 계산            
                cv2.putText(image,f'Degree : {int(angle_to_head_deg)}',(20, 90),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3) # 각도 표시
                if angle_to_head_deg>30:
                    print('회전2')#양수
                #print(angle_to_head_deg)

                cv2.line(image, (l[0], l[1]), (l[2], l[3]), [0, 0, 255], 6)# / 인식된 선 표시 -> 파란색

            return image, angle_to_head_deg, detection

        else : # 선이 인식 되지 않는 경우
            detection = [False, False]

            return image, angle_to_head_deg, detection


if __name__ == "__main__":
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)
    #1111111#
    #설정없이 차이  resize

    while True:
        ###################이미지 기본 셋팅########################
        _, img = cap.read()
        line_detector=ld.lineDetector()
        start_t = timeit.default_timer()

        imgCopy = img.copy()
        imgFiltered = imgFilter.hsv(img)
        imgPreprocessing = imgFilter.preProcessing(imgFiltered)
        ###################라인 그리기######################## 
        lines = cv2.HoughLinesP(imgPreprocessing, 1, np.pi/180, 100, minLineLength = 25, maxLineGap = 10) # HoughLinesP를 이용하여 차선 검출
        left_line, right_line = line_detector.getAverageSlope(imgPreprocessing, lines)
        heading_image, head_angle, detected_line = line_detector.angleToHead(img, left_line, right_line)


        imgBlank = np.zeros_like(img)
        imgStack = stackImages.stack(0.8, ([imgCopy, imgFiltered, imgPreprocessing],
                                     [img, imgBlank, imgBlank]))

        cv2.imshow('Result', imgStack)

        terminate_t = timeit.default_timer()
        FPS = int(1./(terminate_t - start_t ))    
        print(FPS)                      
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






