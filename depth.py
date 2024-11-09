import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from constants import aruco_positions, mtx, dist
import rclpy
import pyrealsense as pyrs

INVERSE_LOCALIZATION = False
DISPLAY = True

class TagPose(Node):
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
tag_node = TagPose()

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)
serv = pyrs.Service()
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60), pyrs.stream.DepthStream(fps = 60)]) # type: ignore

while True:
    cam.wait_for_frames()
    frame = cam.color
    depth = cam.depth
    # d = convert_z16_to_bgr(d)
    print(depth)
    
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

rclpy.shutdown()