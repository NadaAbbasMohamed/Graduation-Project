import cv2
import numpy as np
import glob
import face_recognition as faceRec

# variables initializations:
scale = 0.25

#Blue = (255,0,0)
#Green = (0,255,0)
Red = (0, 0, 255)
white = (255, 255, 255)

illegal = 0
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
det_encoded = []

# load all detected faces stored in the directory into a large tupple: 
#search for another method for reading and storing multiple images
detected = [cv2.imread(file) for file in glob.glob("C:/Users/nada/Desktop/Graduation Project/detected faces/*.jpg")]
inx = 0
for img in detected:
    test = faceRec.face_encodings(img)[0]
    det_encoded.append(test)

# generate the same sequence of the IDs given by the FRID module
#img_encoded_ID =

cap = cv2.VideoCapture(0)
while(1):

    ret, frame = cap.read()
    S_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)     # understand later
    S_frame = S_frame[:, :, ::-1]     # understand later - convert BGR to RBG

    if process_this_frame == True:
        face_locations = faceRec.face_locations(S_frame)
        face_encodings = faceRec.face_encodings(S_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = faceRec.compare_faces(det_encoded, face_encoding)
            name = "Illegal Enterance"
            illegal = illegal+1
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = "Legal Enterance"
                illegal = illegal-1

            face_names.append(name)
            print("Illegal Enterence #: ", illegal)

    process_this_frame = not process_this_frame      # stop processing the frame for delay reduction

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), Red, 2)
                
         # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), Red, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, white, 1)
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()       
