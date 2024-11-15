import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from constants import aruco_position, mtx, dist
import rclpy
import pyrealsense as pyrs
from nav_msgs.msg import OccupancyGrid, MapMetaData

INVERSE_LOCALIZATION = False
DISPLAY = True
GRID_RESOLUTION = 0.01

class Grid(Node):
    def __init__(self):
        super().__init__('grid')
        self.publisher_ = self.create_publisher(OccupancyGrid, '/light_grid', 10)

    def publish(self, img):
        img = ((img / 255) * 127).astype(np.int8)

        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info = MapMetaData()
        msg.info.resolution = GRID_RESOLUTION
        msg.info.width = img.shape[0]
        msg.info.height = img.shape[1]
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.data = img.flatten('F').tolist()
        self.publisher_.publish(msg)

class Pose(Node):
    def __init__(self):
        super().__init__('overthruster') # type: ignore
        self.publisher_ = self.create_publisher(PoseStamped, '/camera_pose', 10)

    def publish(self, coords, rvec):
        msg = PoseStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        orientation = Rot.from_euler('xyz', [0, 0, -1 * Rot.from_matrix(rvec).as_euler('xyz')[2]]).as_quat()
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        
        msg.pose.position.y = float(coords[0] * GRID_RESOLUTION)
        msg.pose.position.x = float(coords[1] * GRID_RESOLUTION)
        msg.pose.position.z = 0.0

        self.publisher_.publish(msg)

# ros setup
rclpy.init()
grid_node = Grid()
pose_node = Pose()

# aruco detector setup
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())

# r200 initialization
serv = pyrs.Service()
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60), ]) # type: ignore

while True:
    cam.wait_for_frames()
    frame = cam.color
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # get corners and display if enabled
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None: cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # flatten corners into usable format, skip if none detected
    if ids is not None and 0 in ids:
        corners = corners[0][0]
        # construct real and camera point comparison matricies
        img_points = []
        real_points = []
        for i in range(0, len(ids)):
            for i in range(0, 4):
                img_points.append(corners[i])
                real_points.append(aruco_position[i])
        real_points = np.array(real_points).astype(np.float32)
        img_points = np.array(img_points).astype(np.float32)

        # solve PnP
        _, rvec, tvec = cv2.solvePnP(real_points, img_points, mtx, dist)
        _, thresh_frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 60, 255, cv2.THRESH_BINARY)
        grid_node.publish(thresh_frame)
        rot = cv2.Rodrigues(rvec)[0]
        pose_node.publish(np.mean(corners, axis=0), rot)

    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

rclpy.shutdown()