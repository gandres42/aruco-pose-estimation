import cv2
import numpy as np
import glob
import os
import math
import time
from calibration import calibrate_pinhole


_, mtx, dist, r_vecs, t_vecs = calibrate_pinhole('./pinhole_images', '', 'jpg', 25)
print(mtx)
print(dist)

cap = cv2.VideoCapture('/dev/video0')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)

while True:
    ret, frame = cap.read()

    corners, ids, rejected = detector.detectMarkers(frame)
    if corners == ():
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        continue
    
    corners = np.array(corners[0])
    obj_points = np.array([[
        [0, 0, 0],
        [180, 0, 0],
        [180, 180, 0],
        [0, 180, 0]
    ]], dtype=np.float32)
    _, rvec, tvec = cv2.solvePnP(obj_points, corners, mtx, dist)

    Rt = cv2.Rodrigues(rvec)[0]
    R = Rt.transpose()
    pos = -R * tvec #type: ignore

    pitch = float(math.atan2(-R[2][1], R[2][2]))
    yaw = math.asin(R[2][0])
    roll = math.atan2(-R[1][0], R[0][0])

    # courtesy of https://www.chiefdelphi.com/t/finding-camera-location-with-solvepnp/159685/6
    ZYX, jac = cv2.Rodrigues(rvec)
    totalrotmax = np.array([[ZYX[0, 0], ZYX[0, 1], ZYX[0, 2], tvec[0][0]], [ZYX[1, 0], ZYX[1, 1], ZYX[1, 2], tvec[1][0]], [ZYX[2, 0], ZYX[2, 1], ZYX[2, 2], tvec[2][0]], [0, 0, 0, 1]])
    WtoC = np.mat(totalrotmax)
    inverserotmax = np.linalg.inv(totalrotmax)

    x = inverserotmax[0][3]
    y = inverserotmax[1][3]
    z = inverserotmax[2][3]

    os.system('cls' if os.name == 'nt' else 'clear')
    print("x: " + str(round(x, 3)))
    print("y: " + str(round(y, 3)))
    print("z: " + str(round(z, 3)))
    print("yaw: " + str(round(math.degrees(yaw), 3)))
    print("pitch: " + str(round(math.degrees(pitch), 3)))
    print("roll: " + str(round(math.degrees(roll), 3)))


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break