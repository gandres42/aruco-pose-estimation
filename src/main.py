#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import filterpy
import open3d as o3d
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import time

class ArucoEstimator:
    def __init__(self, display=True):
        self.cv_bridge = CvBridge()
        
        # read config
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.display = self.config['display']
        self.mtx = np.array(self.config['camera']['mtx'])
        self.dist = np.array(self.config['camera']['dist'])

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
        self.f.P *= 5.0

        self.prev_f_time = rospy.Time.now().to_nsec()

        # subscribe to video
        self.image_sub = rospy.Subscriber('/BlueROV2/video', Image, self.cam_cb, queue_size=1)

        # publish annotated image and pose
        self.annotated_pub = rospy.Publisher('/aruco/annotated', Image, queue_size=1)
        self.pose_pub = rospy.Publisher('/aruco/pose', PoseStamped, queue_size=1)

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
        # corners, ids, rejected = self.detector.detectMarkers(frame)
        if str(cv2.__version__) != '4.2.0':
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            detectorParams = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            detectorParams = cv2.aruco.DetectorParameters_create()
            corners, ids, rejected = cv2.aruco.detectMarkers(
                frame, dictionary, parameters=detectorParams
            )

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
        dt = (rospy.Time.now().to_nsec() - self.prev_f_time) * 1e-9
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
        self.prev_f_time = rospy.Time.now().to_nsec()

        # publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'base_link'  # or use your world frame name
        pose_msg.pose.position.x = float(self.f.x[0])
        pose_msg.pose.position.y = float(self.f.x[1])
        pose_msg.pose.position.z = float(self.f.x[2])
        self.pose_pub.publish(pose_msg)


def main():
    rospy.init_node('aruco_node')
    estimator = ArucoEstimator()
    rospy.spin()

if __name__ == "__main__":
    main()