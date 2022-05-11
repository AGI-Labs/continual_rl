"""
From https://github.com/AGI-Labs/franka_control/blob/b369bb1bc4d83ebe10d493f253bfbace2cc585ef/camera.py
"""

import cv2
import pyrealsense2 as rs
import numpy as np


class Camera:
    def __init__(self):
        # initialize camera
        self._connect_cam()
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image

        color_frame = aligned_frames.get_color_frame()
        depth_image = self._resize(np.asanyarray(aligned_depth_frame.get_data()))
        color_image = self._resize(np.asanyarray(color_frame.get_data()))

    def _resize(self, img):
        return cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)

    def _connect_cam(self):
        self.pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start streaming and create frame aligner
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        for _ in range(100):
            frames = self.pipeline.wait_for_frames()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        depth_image = self._resize(np.asanyarray(aligned_depth_frame.get_data()))
        color_image = self._resize(np.asanyarray(color_frame.get_data()))
        return color_image, depth_image
