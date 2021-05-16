import cv2
import glob
import face_recognition as faceRec

# function declarations:

# -1-  Bounding Box Function:
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

# -2- Recognition function:
def Recognition_Refresh(img, detected):

    det_encoded = []
    face_locations = []
    face_encodings = []
    matches = []
    scale = 0.25 
    # encoding detected images:
    for img in detected:
        test = faceRec.face_encodings(img)[0]
        det_encoded.append(test)
        
   # prepare frame for detection and recognition:     
    S_frame = cv2.resize(img, (0, 0), fx = scale, fy = scale)     # understand later
    S_frame = S_frame[:, :, ::-1]                                               # understand later - convert BGR to RBG

    # comparing captured face with previously stored ones:
    if process_this_frame == True:
        face_locations = faceRec.face_locations(S_frame)
        face_encodings = faceRec.face_encodings(S_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = faceRec.compare_faces(det_encoded, face_encoding)
            # If a match was found in known_face_encodings, just use the first one.

    return matches, face_locations
# -3- Define centroids of faces detected:
def centroid_def(dimensions):
    centers=[]
    for (x1,y1,x2,y2) in dimensions:
        Cx = int((x1+x2)/2.0)               # this might be wrong if considering x2 and y2 to be just distances
        Cy = int((y1+y2)/2.0)
        centers.append([Cx,Cy])
    return centers

# -3- Tracking:
#def face_tracking():

# -4- Test Entered:
def test_entered(prev_count):
    detected = [cv2.imread(file) for file in glob.glob("C:/Users/nada/Desktop/Graduation Project/detected faces/*.jpg")]
    count = len(detected)
    if count> prev_count:
        entered = True
    else:
        entered = False
    return detected, entered, count
#########################################################################################################
# MAIN FUNCTION:

# Variable Initializations:
Blue = (255, 0, 0)
Green = (0, 255, 0)
Red = (0, 0, 255)
white = (255, 255, 255)
black = (0, 0, 0)
illegal = 0
num_entered = 0
inx = 0
entered = False    # change from false to true if a new face is detected and saved from camera 1
new_matches = []
face_boundaries = []
face_names = []
process_this_frame = True

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    detected, entered, num_entered = test_entered(num_entered)
    if entered == True:
        new_matches, face_boundaries = Recognition_Refresh(frame, detected)

    if True in new_matches:
        first_match_index = new_matches.index(True)
        name = "Legal Entrance"
    else:
        name = "Illegal Entrance"
        illegal = illegal + 1

    face_names.append(name)
    process_this_frame = not process_this_frame
    entered = False
    cv2.putText(frame, "illegal Entrances: " + str(illegal), (frame.shape[1] - 180, frame.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, black, 1)
    Bound_Box(face_boundaries, face_names, frame, Green)
    centroid_def(face_boundaries)
#    track()  

        





















        
                        
