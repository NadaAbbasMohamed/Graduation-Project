import numpy as np
import cv2
import glob

# change corners number with 6*9 for default openCV samples

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

ref = cv2.imread("fig 22.jpg")
dim = (ref.shape[1], ref.shape[0])

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, dim,  interpolation = cv2.INTER_CUBIC)   
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 7), corners2,ret)
        cv2.imshow(''+fname,img)
        cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[: :-1],None,None)
# rvecs: rotation vectors - tvecs: translation vectors
# refine the camera matrix based on a free scaling parameter:
img = cv2.imread('fig 28.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# method -1- :
dst1 = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst1 = dst1[y:y+h, x:x+w]
dst1 = cv2.resize(dst1, dim,  interpolation = cv2.INTER_CUBIC)  
cv2.imshow('RESULT REQUIRED - 1' ,dst1)

# method -2- :
#mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst2 = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
#x,y,w,h = roi
#dst2 = dst2[y:y+h, x:x+w]
#cv2.imshow('RESULT REQUIRED - 2' ,dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()

print( "new camera matrix: \n", newcameramtx)
print( " \ndistribution parameters: \n",  dist)
