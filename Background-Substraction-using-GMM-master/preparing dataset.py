import numpy as np
import cv2
import os
import random
import pickle

# Preparing Data Set:
data_dir = 'C:/Users/nada/Desktop/Gaussian Mixture Model - Algorithm Papers/dataset/train'
face_csc = cv2.CascadeClassifier('C:/OpenCV/opencv/sources/data/haarcascades/haarcascade_fullbody.xml')
CATEGORIES = ["human", "No_human"]
training_data = []
img_size = 224

for CATEGORY in CATEGORIES:
	path = os.path.join(data_dir, CATEGORY)
	class_num = CATEGORIES.index(CATEGORY) 
	for img in os.listdir(path):
		try:			
			img = cv2.imread(os.path.join(path, img))
			if(CATEGORY == "human"):
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				faces = face_csc.detectMultiScale(gray)
				img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				for (x, y, w, h) in faces:
					cropped_ROI = img[y:y+h, x:x+h]
					new_array = cv2.resize(cropped_ROI, (img_size,img_size), interpolation = cv2.INTER_AREA)

			else:
				new_array = cv2.resize(img, (img_size,img_size) ,interpolation = cv2.INTER_AREA)	
			training_data.append([new_array, class_num])
		except Exception as e:
			pass
 
random.shuffle(training_data)

# Convert images to appropraite representation: labels and features:
x_train = []	# Feature Set - images 
y_train = []	# Label Set

for features, label in training_data:
	x_train.append(features)
	y_train.append(label)

x_train =np.array(x_train)#.reshape(-1, img_size,img_size,3)	
# -1: to get all total number of images - kol l images eli fe l array maisebsh 7aga

pickle_out = open("x_train.pickle", "wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()  		

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close() 


# Preparing Data Set:
data_dir = 'C:/Users/nada/Desktop/Gaussian Mixture Model - Algorithm Papers/dataset/test'
CATEGORIES = ["human", "No_human"]
training_data = []
img_size = 224
for CATEGORY in CATEGORIES:
	path = os.path.join(data_dir, CATEGORY)
	class_num = CATEGORIES.index(CATEGORY) 
	for img in os.listdir(path):
		try:			
			img = cv2.imread(os.path.join(path, img))
			if(CATEGORY == "human"):
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				faces = face_csc.detectMultiScale(gray)
				img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
				for (x, y, w, h) in faces:
					cropped_ROI = img[y:y+h, x:x+h]
					new_array = cv2.resize(cropped_ROI, (img_size,img_size), interpolation = cv2.INTER_AREA)

			else:
				new_array = cv2.resize(img, (img_size,img_size) ,interpolation = cv2.INTER_AREA)	
			training_data.append([new_array, class_num])
		except Exception as e:
			pass
 
random.shuffle(training_data)

# Convert images to appropraite representation: labels and features:
x_test = []	# Feature Set - images 
y_test = []	# Label Set

for features, label in training_data:
	x_test.append(features)
	y_test.append(label)

x_test =np.array(x_test)	# -1: to get all total number of images - kol l images eli fe l array maisebsh 7aga

pickle_out = open("x_test.pickle", "wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()  		

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()  	