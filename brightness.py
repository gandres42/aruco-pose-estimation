import cv2
import numpy as np
import pyrealsense as pyrs
from pyrealsense.constants import rs_option # type: ignore
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData

class Grid(Node):
    def __init__(self):
        super().__init__('grid') # type: ignore
        self.publisher_ = self.create_publisher(OccupancyGrid, '/light_grid', 10)

    def publish(self,img):
        msg = OccupancyGrid()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()

        # Fill in metadata (MapMetaData)
        msg.info = MapMetaData()
        msg.info.resolution = 0.01  # Each cell is 10cm x 10cm
        msg.info.width = img.shape[0] # Width of the grid in cells
        msg.info.height = img.shape[1] # Height of the grid in cells
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0

        # Fill in data
        print(len([-1] * (msg.info.width * msg.info.height)))
        # print(img.flatten().as_list())
        msg.data = img.flatten('F').tolist()
        

        self.publisher_.publish(msg)


serv = pyrs.Service()
custom_options = [(rs_option.RS_OPTION_COLOR_EXPOSURE, 156), (rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 0)]
cam = serv.Device(device_id = 0, streams = [pyrs.stream.ColorStream(fps = 60), ]) # type: ignore
cam.set_device_options(*zip(*custom_options))

rclpy.init()
node = Grid()

while True:
    cam.wait_for_frames()
    frame = cam.color
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    noir_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(noir_frame, 60, 255, cv2.THRESH_BINARY)

    new_thresh = ((thresh / 255) * 127).astype(np.int8)
    node.publish(new_thresh)

    cv2.imshow('Camera', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
rclpy.shutdown()