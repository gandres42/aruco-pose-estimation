import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from constants import aruco_positions, mtx, dist
import math
import rclpy

INVERSE_LOCALIZATION = False
DISPLAY = True

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

    def listener_callback(self, msg):
        position = (np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]) @ self.init_orientation) + self.init_position # type: ignore
        orientation = Rot.from_matrix(
            Rot.from_quat([
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w
            ]).as_matrix() @ self.init_orientation
        ).as_quat() # type: ignore

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

class Localization(Node):
    def __init__(self):
        super().__init__('overthruster') # type: ignore
        self.publisher_ = self.create_publisher(PoseStamped, '/camera_pose', 10)

    def publish(self, Rt, position):
        msg = PoseStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        orientation = Rot.from_matrix(Rt).as_quat()
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        
        msg.pose.position.x = position[2] / 10
        msg.pose.position.y = position[0] / -10
        msg.pose.position.z = position[1] / 10

        self.publisher_.publish(msg)


rclpy.init()

node = Localization()

cap = cv2.VideoCapture('/dev/video4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_MODE, 5)

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    # noir_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)
    
    # get corners and display if enabled
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None: cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    
    # flatten corners into usable format, skip if none detected
    if corners != ():
        flat_corners = ids.flatten().tolist()
        if not (flat_corners != [0, 1] and flat_corners != [1, 0] and flat_corners != [0]):
            # construct real and camera point comparison matricies
            img_points = []
            real_points = []
            for i in range(0, len(ids)):
                for i in range(0, 4):
                    img_points.append(corners[0][0][i])
                    real_points.append(aruco_positions[0][0][i])
            real_points = np.array(real_points).astype(np.float32)
            img_points = np.array(img_points).astype(np.float32)

            # solve PnP
            _, rvec, tvec = cv2.solvePnP(real_points, img_points, mtx, dist)

            # transform into world frame
            rot = cv2.Rodrigues(rvec)[0]
            
            # inverse to camera localization is specified
            if INVERSE_LOCALIZATION:
                R = rot.transpose()
                pos = -R * tvec
                ZYX, jac = cv2.Rodrigues(rvec)
                totalrotmax = np.array([[ZYX[0, 0], ZYX[0, 1], ZYX[0, 2], tvec[0][0]], [ZYX[1, 0], ZYX[1, 1], ZYX[1, 2], tvec[1][0]], [ZYX[2, 0], ZYX[2, 1], ZYX[2, 2], tvec[2][0]], [0, 0, 0, 1]])
                rot = np.linalg.inv(totalrotmax)
                
            if DISPLAY:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = .5
                color = (255, 255, 255)  # White color for text
                thickness = 1
                position = (10, frame.shape[0] - 10)
                ypr = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
                yaw = ypr[1]
                pitch = ypr[0]
                roll = ypr[2]
                x = tvec[0, 0]
                y = tvec[1, 0]
                z = tvec[2, 0]
                cv2.putText(frame, f"x: {x:.2f}", (0, frame.shape[0] - 20), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"y: {y:.2f}", (120, frame.shape[0] - 20), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"z: {z:.2f}", (240, frame.shape[0] - 20), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"yaw: {yaw:.2f}", (0, frame.shape[0] - 5), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"pitch: {pitch:.2f}", (120, frame.shape[0] - 5), font, font_scale, color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"roll: {roll:.2f}", (240, frame.shape[0] - 5), font, font_scale, color, thickness, cv2.LINE_AA)
            
            node.publish(rot, tvec.flatten())

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

# Start pose transformation node
# rclpy.init()

rclpy.shutdown()