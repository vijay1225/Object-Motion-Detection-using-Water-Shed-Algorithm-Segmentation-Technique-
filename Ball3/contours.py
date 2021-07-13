import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot
import time

i = 1
data = cv2.VideoCapture('ball3.mp4')
ret,frame1 = data.read()

ret,frame2 = data.read()

#t1=time.time()
while data.isOpened():
    #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    #gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(frame1, frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _, thresh =cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=10)
    _, contours, _ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #
   # for contour in contours:
   #     (x,y,w,h)=cv2.boundingRect(contour)
    #       continue
     #       ccv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
      #      cv2.putText(frame1,"Status:{}".format('Movement'),(10,20),cv2.FONT_HERSHY_SIMPLEX,1,(0,0,255),3)





    cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    #t2=time.time()
    cv2.imshow("feed",frame1)
   # cv2.imwrite("C:/Users/raja/Desktop/project2/jerry1_contours/duplicate_color_contours" + str( i) + '.png', frame1)
    frame1=frame2.copy()
    ret,frame2=data.read()
    #pyplot.imshow(frame2)
    #pyplot.show()

    i += 1
    if cv2.waitKey(100)==27:
         break
cv2.destroyAllWindows()
data.release()