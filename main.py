import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from constants import aruco_positions, mtx, dist
import rclpy
import pyrealsense as pyrs
from nav_msgs.msg import OccupancyGrid, MapMetaData

INVERSE_LOCALIZATION = False
DISPLAY = True

class Grid(Node):
    def __init__(self):
        super().__init__('grid') # type: ignore
        self.publisher_ = self.create_publisher(OccupancyGrid, '/light_grid', 10)

    def publish(self, img, cx, cy):
        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        # Fill in metadata (MapMetaData)
        msg.info = MapMetaData()
        msg.info.resolution = 0.01  # Each cell is 10cm x 10cm
        msg.info.width = img.shape[0] # Width of the grid in cells
        msg.info.height = img.shape[1] # Height of the grid in cells
        msg.info.origin.position.x = float(cy) * -.01
        msg.info.origin.position.y = float(cx) * -.01
        msg.info.origin.position.z = 0.0
        msg.data = img.flatten('F').tolist()
        self.publisher_.publish(msg)


rclpy.init()
mcgriddle = Grid()

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.DetectorParameters()
)
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

            # find homographic transform for frame
            tag_size = 100  # desired size of the ArUco tag in the output image
            points_dst = np.array([[0, 0], [tag_size, 0], [tag_size, tag_size], [0, tag_size]], dtype='float32')
            h_matrix, _ = cv2.findHomography(img_points, points_dst)
            height, width = frame.shape[:2]
            image_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype='float32')
            transformed_corners = cv2.perspectiveTransform(np.array([image_corners]), h_matrix)[0]
            [x_min, y_min] = np.int32(transformed_corners.min(axis=0)) # type: ignore
            [x_max, y_max] = np.int32(transformed_corners.max(axis=0)) # type: ignore
            output_width, output_height = x_max - x_min, y_max - y_min
            translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
            adjusted_h_matrix = translation_matrix @ h_matrix
            frame = cv2.warpPerspective(frame, adjusted_h_matrix, (output_width, output_height)) # type: ignore

            new_corners, new_ids, _ = detector.detectMarkers(frame)
            if new_ids is not None and 0 in new_ids[0]:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, frame = cv2.threshold(frame, 60, 255, cv2.THRESH_BINARY)
                grid = ((frame / 255) * 127).astype(np.int8)
                mcgriddle.publish(grid, np.mean(new_corners[0][0][:, 0]), np.mean(new_corners[0][0][:, 1]))

    # cv2.imshow('Camera', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

rclpy.shutdown()