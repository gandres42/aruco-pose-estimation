import cv2
import numpy as np
import math
from calibration import calibrate_pinhole
from scipy.spatial.transform import Rotation as Rot
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import subprocess
import re

def get_camera_info():
    endpoints = subprocess.run(("ls", "/dev"), capture_output=True).stdout.decode().split('\n')
    cameras = [f"/dev/{x}" for x in endpoints if re.compile('video\d+').match(x)]
    cams = []
    for camera in cameras:
        lines = subprocess.run(("v4l2-ctl", "-d", camera, "--info"), capture_output=True).stdout.decode().splitlines()
        for line in lines:
            if 'Serial' in line:
                _, serial = line.replace('\t', '').replace(' ', '').split(':')
                cams.append((serial, camera))
                break
    return tuple(cams)


class Transformer(Node):
    def __init__(self, init_position, init_orientation):
        super().__init__('overthruster') # type: ignore
        self.init_position = init_position
        self.init_orientation = init_orientation
        self.subscription = self.create_subscription(
            PoseStamped,
            '/dlio/odom_node/pose',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(PoseStamped, '/adjusted_pose', 10)
        print("delivered")

    def listener_callback(self, msg):
        position = (np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]) @ self.init_orientation) + self.init_position
        orientation = Rot.from_matrix(
            Rot.from_quat([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]).as_matrix() @ self.init_orientation
        ).as_quat()


        new_msg = PoseStamped()

        new_msg.header = msg.header
        new_msg.header.stamp = self.get_clock().now().to_msg()

        new_msg.pose.orientation.x = orientation[0]
        new_msg.pose.orientation.y = orientation[1]
        new_msg.pose.orientation.z = orientation[2]
        new_msg.pose.orientation.w = orientation[3]
        
        new_msg.pose.position.x = position[0]
        new_msg.pose.position.y = position[1]
        new_msg.pose.position.z = position[2]

        self.publisher_.publish(new_msg)

class ArucoEstimator():
    def __init__(self):
        pass

    def camera_thead()

positions = [
    np.array([[
        [10, 0, 0],
        [190, 0, 0],
        [190, 180, 0],
        [10, 180, 0]
    ]]),
    np.array([[
        [0, 0, 190],
        [0, 0, 10],
        [0, 180, 10],
        [0, 180, 190]
    ]]),
]
_, mtx, dist, r_vecs, t_vecs = calibrate_pinhole('./pinhole_images', '', 'jpg', 25)

cap = cv2.VideoCapture('/dev/video0')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)

roll = None
pitch = None
yaw = None
x = None
y = None
z = None

print("sined...")
while roll is None:
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    
    corners, ids, rejected = detector.detectMarkers(frame)
    if corners == ():
        continue
    flat_corners = ids.flatten().tolist()
    if flat_corners != [0, 1] and flat_corners != [1, 0] and flat_corners != [0]:
        continue

    img_points = []
    real_points = []

    for i in range(0, len(ids)):
        for i in range(0, 4):
            img_points.append(corners[0][0][i])
            real_points.append(positions[0][0][i])

    real_points = np.array(real_points).astype(np.float32)
    img_points = np.array(img_points).astype(np.float32)

    _, rvec, tvec = cv2.solvePnP(real_points, img_points, mtx, dist)

    Rt = cv2.Rodrigues(rvec)[0]
    R = Rt.transpose()
    pos = -R * tvec #type: ignore

    ZYX, jac = cv2.Rodrigues(rvec)
    totalrotmax = np.array([[ZYX[0, 0], ZYX[0, 1], ZYX[0, 2], tvec[0][0]], [ZYX[1, 0], ZYX[1, 1], ZYX[1, 2], tvec[1][0]], [ZYX[2, 0], ZYX[2, 1], ZYX[2, 2], tvec[2][0]], [0, 0, 0, 1]])
    WtoC = np.mat(totalrotmax)
    inverserotmax = np.linalg.inv(totalrotmax)

    pitch = float(math.atan2(-R[2][1], R[2][2]))
    yaw = math.asin(R[2][0])
    roll = math.atan2(-R[1][0], R[0][0])
    x = inverserotmax[0][3]
    y = inverserotmax[1][3]
    z = inverserotmax[2][3]
    cv2.destroyAllWindows()
    cap.release()


init_position = np.array([x, y, z])
init_orientation = Rot.from_euler('xyz', (roll, pitch, yaw), degrees=True).as_matrix() # type: ignore

print("seeled...")
rclpy.init()
minimal_subscriber = Transformer(init_position, init_orientation)
rclpy.spin(minimal_subscriber)
minimal_subscriber.destroy_node()
rclpy.shutdown()