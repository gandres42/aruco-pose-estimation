import cv2
from linuxpy.video.device import Device
import subprocess, re
import math
        

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
