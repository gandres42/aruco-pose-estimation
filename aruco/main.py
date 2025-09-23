import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import filterpy
import open3d as o3d
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class ArucoEstimator(Node):
    def __init__(self, display=True):
        super().__init__('aruco_node')
        self.cv_bridge = CvBridge()
        
        # read config
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.display = self.config['display']
        self.mtx = np.array(self.config['camera']['mtx'])
        self.dist = np.array(self.config['camera']['dist'])
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
            cv2.aruco.DetectorParameters()
        )

        # generate aruco dictionary
        self.tags = {}
        pcds = []
        for tag_id, tag_dict in self.config['tags'].items():
            d = tag_dict['size'] / 2
            center_p = np.array(tag_dict['center'])
            corner_points = np.array([
                np.array(center_p) + np.array([d, d, 0]),
                np.array(center_p) + np.array([d, -d, 0]),
                np.array(center_p) + np.array([-d, -d, 0]),
                np.array(center_p) + np.array([-d, d, 0])
            ])
            tag_R = R.from_euler('xyz', tag_dict['rotation']).as_matrix()
            corner_points = ((corner_points - center_p) @ tag_R) + center_p
            
            self.tags[int(tag_id)] = corner_points
            # pcd = o3d.geometry.PointCloud()
            # colors = np.array([
            #     [1, 0, 0],   # red
            #     [0, 1, 0],   # green
            #     [0, 0, 1],   # blue
            #     [0, 0, 0]    # black
            # ])
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd.points = o3d.utility.Vector3dVector(corner_points)
            # pcds.append(pcd)
        
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # pcds.append(axis)
        # o3d.visualization.draw_geometries(pcds)
        
        # kalman filter
        self.f = KalmanFilter(dim_x=6, dim_z=3)
        self.f.x = np.array([
            [0],
            [0],
            [0],
            [0],
            [0],
            [0]
        ])
        self.f.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        self.f.P *= 10.0
        # self.f.R = 10.0

        self.prev_f_time = self.get_clock().now().nanoseconds

        # subscribe to video
        self.create_subscription(Image, '/BlueROV2/video', self.cam_cb, 1)

        # publish annotated image and pose
        self.annotated_pub = self.create_publisher(Image, '/aruco/annotated', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 1)

    def make_Q(self, dt, sigma_a):
        q = sigma_a**2
        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4

        Q = np.array([
            [dt4/4,    0,       0,   dt3/2,    0,      0],
            [0,    dt4/4,       0,      0,  dt3/2,     0],
            [0,       0,    dt4/4,      0,     0,   dt3/2],
            [dt3/2,   0,       0,    dt2,     0,      0],
            [0,    dt3/2,      0,      0,   dt2,      0],
            [0,       0,    dt3/2,     0,     0,    dt2]
        ]) * q
        return Q
    
    def cam_cb(self, msg):
        # tag identification and display
        frame = self.cv_bridge.imgmsg_to_cv2(msg)
        corners, ids, rejected = self.detector.detectMarkers(frame)

        if self.display:
            annotated_frame = frame.copy()
            if ids is not None and len(corners) > 0:
                cv2.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
                annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                self.annotated_pub.publish(annotated_msg)
            else:
                annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                self.annotated_pub.publish(annotated_msg)
        if ids is None: return

        # create matched set of image and real points
        img_pts = []
        tag_pts = []
        for i in range(len(ids.flatten())):
            tag_id = ids.flatten()[i]
            if tag_id in self.tags:            
                img_pts.append(corners[i][0])
                tag_pts.append(self.tags[tag_id])
        if len(img_pts) <= 0 or len(tag_pts) <= 0: return

        # Use solvePnP to estimate camera pose
        img_pts_np = np.array(img_pts).reshape(-1, 2)
        tag_pts_np = np.array(tag_pts).reshape(-1, 3)
        success, rvec, tvec = cv2.solvePnP(
            tag_pts_np,
            img_pts_np,
            self.mtx,
            self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success: return

        # convert camera pose to ROS coordinates
        rot_mat, _ = cv2.Rodrigues(rvec)
        T_cv_to_ros = np.array([
            [0, 0, -1],  # x_ros = z_cv
            [1, 0, 0], # y_ros = -x_cv
            [0, 1, 0]  # z_ros = -y_cv
        ])
        rot_ros = T_cv_to_ros @ rot_mat
        tvec_ros = T_cv_to_ros @ tvec

        # kalman update
        dt = (self.get_clock().now().nanoseconds - self.prev_f_time) * 1e-9
        self.f.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        # self.f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.13)
        self.f.Q = self.make_Q(dt, 0.5)
        self.f.predict()
        self.f.update(tvec_ros)
        print(self.f.x)
        self.prev_f_time = self.get_clock().now().nanoseconds

        # publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'  # or use your world frame name
        pose_msg.pose.position.x = float(self.f.x[0])
        pose_msg.pose.position.y = float(self.f.x[1])
        pose_msg.pose.position.z = float(self.f.x[2])
        # Convert rotation matrix to quaternion
        # quat = R.from_matrix(rot_ros).as_quat()  # [x, y, z, w]
        # pose_msg.pose.orientation.x = float(quat[0])
        # pose_msg.pose.orientation.y = float(quat[1])
        # pose_msg.pose.orientation.z = float(quat[2])
        # pose_msg.pose.orientation.w = float(quat[3])
        self.pose_pub.publish(pose_msg)

        


def main(args=None):
    rclpy.init(args=args)
    node = ArucoEstimator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()