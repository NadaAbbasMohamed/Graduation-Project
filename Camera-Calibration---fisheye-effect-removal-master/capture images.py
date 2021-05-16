import cv2
import numpy as np
import time

cap = cv2.VideoCapture('rtsp://admin:FAR98AWAY$@169.254.146.120/1')

k = 0
while(1):
    frame, ret = cap.read()
  #  cv2.putText(frame, k, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType=cv2.LINE_AA)
    cv2.imshow('frames', frame)
    k = k+1
    time.sleep(3)
    if k == 10:
        break        

#i = 30 
#while(1):
#    ret, frame = cap.read()
#    cv2.imshow('frames', frame)
#    cv2.imwrite('Img'+i+'.jpg', frame)
#    time.sleep(5)
#    i = i-1
#    if i==0:
#        break
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break

cap.release()
cv2.destroyAllWindows()
#cap.release()
#cv2.destroyAllWindows()
