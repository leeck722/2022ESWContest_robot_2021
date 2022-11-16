import cv2
import math
import numpy as np
import imgFilter
import stackImages
import timeit
import lineDetect as ld

if __name__ == "__main__":
    frameWidth = 640
    frameHeight = 480
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)

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

        terminate_t = timeit.default_timer()



        FPS = int(1./(terminate_t - start_t ))    
        #print(FPS)                      
        cv2.imshow('Result', imgStack)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




cap.release()
cv2.destroyAllWindows()