import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from constants import aruco_positions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ArucoEstimator(Node):
    def __init__(self, display=True):
        super().__init__('aruco_node')
        self.display = display

        # camera calibration
        self.mtx = np.array([
            [1078.17559, 0.0, 1010.57086],
            [0.0, 1076.46176, 463.06243],
            [0.0, 0.0, 1.0]
        ])
        self.dist = np.array([0.019645, 0.007271, -0.004324, -0.001628, 0.000000])
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
            cv2.aruco.DetectorParameters()
        )

        self.create_subscription(Image, '/BlueROV2/video', self.cam_cb, 1)
        self.cv_bridge = CvBridge()
        self.annotated_pub = self.create_publisher(Image, '/aruco/annotated', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 1)
    
    def cam_cb(self, msg):
        # region tag identification and display
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
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
        print(ids)
        if ids is None: return
        # endregion

        # region pose estimation
        img_points = []
        real_points = []
        for tag_id in ids[0]:
            if tag_id in aruco_positions:
                for i in range(0, 4):
                    img_points.append(corners[0][0][i])
                    real_points.append(aruco_positions[tag_id][i])
        if len(real_points) < 4: return

        real_points = np.array(real_points).astype(np.float32)
        img_points = np.array(img_points).astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(real_points, img_points, self.mtx, self.dist)

        print(tvec)

        Rt = cv2.Rodrigues(rvec)[0]
        R = Rt.transpose()
        # endregion

        tvec = tvec.flatten()
        
        # region publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = tvec[0] / 100
        pose_msg.pose.position.y = tvec[1] / 100
        pose_msg.pose.position.z = tvec[2] / 100
        # quat = Rot.from_matrix(R).as_quat()
        # pose_msg.pose.orientation.x = quat[0]
        # pose_msg.pose.orientation.y = quat[1]
        # pose_msg.pose.orientation.z = quat[2]
        # pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # endregion

        return R, tvec



def main(args=None):
    rclpy.init(args=args)
    node = ArucoEstimator()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()