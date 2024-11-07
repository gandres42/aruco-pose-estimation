import cv2
import numpy as np
import pyrealsense as pyrs
from pyrealsense.constants import rs_option # type: ignore

cv2.namedWindow("preview")

serv = pyrs.Service()
custom_options = [(rs_option.RS_OPTION_COLOR_EXPOSURE, 156), (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0)]
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60), ]) # type: ignore
cam.set_device_options(*zip(*custom_options))

while True:
    cam.wait_for_frames()
    frame = cam.color
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    noir_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(noir_frame, 60, 255, cv2.THRESH_BINARY)
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