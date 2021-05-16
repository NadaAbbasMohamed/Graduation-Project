# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 01:20:13 2020

@author: nada
"""
import os
import cv2
import time
import numpy as np
import keras
from keras.models import model_from_json
from keras.preprocessing import image

cap = cv2.VideoCapture("People Walking 3.mp4")
#cap = cv2.VideoCapture(0)

# back ground subtraction for object detection - GMM Motion Detection Method: 
fgbg = cv2.createBackgroundSubtractorMOG2()
#model = keras.models.load_model("AlexNet.model")
model = model_from_json(open("fer.json","r").read())
model.load_weights('Weights_AlexNet.h5')
cropped_people = []
img_size = 224

frame_count =0
fps=0
start = time.time()
while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame) #Result of GMM Method
    fgmask = cv2.medianBlur(fgmask ,7)  #Frame Result denoising
  
    im, cont,h = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in cont:
        area = cv2.contourArea(c)

        if area > 300:
            (x, y, w, h) = cv2.boundingRect(c)
            cropped_ROI = frame[y:y+h, x:x+h]
            new_array = cv2.resize(cropped_ROI,(img_size, img_size),interpolation = cv2.INTER_AREA)
            #cv2.imshow('res', new_array)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            res =np.array(new_array).reshape(-1, img_size,img_size, 3)
            predictions = model.predict(res)
            print(predictions)
            if predictions == 1:
                # apply correlation filter tracking on the object
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                #cv2.rectangle(frame, (x, y+h-15),(x+w, y+h),(0, 0, 255), cv2.FILLED)
                #font = cv2.FONT_HERSHEY_DUPLEX
                #cv2.putText(frame, "Person", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            else:
                # no human is detected
                # donot process/ track the object
                continue

    frame_count = frame_count+1
    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
end = time.time()
time = end-start
fps = frame_count/time
print("frames per second: ")
print(fps)
## problems:
# if person is static for number of frames = n it is not detected
# this type of filter cause high noise in the frame (overcome)
# it cannot detect dark colored clothes - not highly reliable as it won't detect hair from above camera