import cv2
import numpy as np

cv2.namedWindow("preview")
vc = cv2.VideoCapture('/dev/video6')

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, frame = vc.read()
    noir_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(noir_frame, 50, 255, cv2.THRESH_BINARY)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # #set the lower and upper bounds for the green hue
    # lower_white = np.array([50,50,50])
    # upper_white = np.array([255,255,255])

    # #create a mask for green colour using inRange function
    # mask = cv2.inRange(hsv, lower_white, upper_white)

    #perform bitwise and on the original image arrays using the mask
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("preview", thresh)
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()