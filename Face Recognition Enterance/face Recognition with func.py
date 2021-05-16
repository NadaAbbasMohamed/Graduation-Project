import cv2
import numpy as np
import glob
import face_recognition as faceRec

# functions declarations:

# Bounding Box function:
def Bound_Box(bound_box, lable, img, color):

    for (top, right, bottom, left), name in zip(bound_box, lable):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 0.5, white, 1)

    cv2.imshow('Video', img)


# variables initializations:

Blue = (255, 0, 0)
Green = (0, 255, 0)
Red = (0, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)

scale = 0.25
illegal = 0
inx = 0

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
det_encoded = []

# load all detected faces stored in the directory into a large tupple:
#search for another method for reading and storing multiple images
detected = [cv2.imread(file) for file in glob.glob("C:/Users/nada/Desktop/Graduation Project/detected faces/*.jpg")]
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

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = "Legal Entrance"
            else:
                name = "Illegal Entrance"
                illegal = illegal + 1

            face_names.append(name)

    process_this_frame = not process_this_frame
    cv2.putText(frame, "illegal Entrances: " + str(illegal), (frame.shape[1] - 180, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, black, 1)
    Bound_Box(face_locations, face_names, frame, Green)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


