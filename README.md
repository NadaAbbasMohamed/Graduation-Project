# Graduation-Project: 
## Indoor security system- integration of RADAR, RFID, and Camera modules for detecting, tracking, and counting number of people in a closed room

The codes of the computer vision/ camera module are made up of 4 parts:
## Camera Calibration for fish eye distortion removal:
we were using in our project a 360 degree camera to achieve full view of the room. this camera generated a curvature in the image resulted known as fish eye distortion.

Camera calibration is a technique used in multiple applications. in this project I used it to calculate the camera matrix (rotational and transitional parameters of the camera) to generate linear view of the monitored are and then enable us from performing further operations on the captured image.
The folder contains 2 files: The 1st is for generating the camera matrix and applying the unistorting operation on a single image The 2nd is for capturing images in a successive sequence as multiple number of chess-board images are required to calculate the camera matrix

is added a sample image showing the fisheye effect on the captured image and the chess board pattern used 8x8

## Face Recognition algorithm:
used at the door enterence to block unwanted/ suspicios people based on a predefined list of blocked people

## Human detection and counting using yolo v3 model

## Integration-Python-with-MATLAB
This is a step by step in my way to integrate our python module project (3 Python codes imported together - ML based for human recognition and detection) into our MATLAB module (RADAR and RFID programs for tracking) in an aim to integrate the 3 outputs onto a single screen -GUI Interface Check the wiki of the project for the project operation sequence

## Other algorithms used and tested for the purpose of the project:
### 1- Background-Substraction-using-GMM:
This include implementation of Background Substraction using Gaussian mixture model. For this, i followed the research paper of Thierry Bouwmans on Background Modelling. I have also implemented this using OpenCV library and then compared both of them. The algorithm of both method and comparison between them is shown in pdf attached with it. Both of the code is written in python.
### 2- Frame subtraction for motion detection:
### 3- 
