#!/usr/bin/env python3

import rclpy
import time
import cv2
import numpy as np

import hello_helpers.hello_misc as hm
import d405_helpers as dh
import aruco_detector as ad


class MotionLoopNode(hm.HelloNode):
    def __init__(self):
        super().__init__()

        # Define only the 3 ArUco markers you care about (no YAML)
        self.allowed_ids = {100, 101, 102}
        self.marker_info = {
            str(i): {'length_mm': 50, 'use_rgb_only': True} for i in self.allowed_ids
        }
        self.marker_info['default'] = {'length_mm': 50, 'use_rgb_only': True}

        # Initialize ArUco detector (using fake marker_info)
        self.aruco_detector = ad.ArucoDetector(
            marker_info=self.marker_info,
            show_debug_images=False,
            brighten_images=True
        )

        self.pipeline, self.profile = dh.start_d405(exposure='low')
        self.first_frame = True
        self.camera_info = None

        # Start ROS 2 HelloNode
        hm.HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node is ready. Starting marker detection...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        self.timer = self.create_timer(0.5, self.motion_loop)

    def motion_loop(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            self.get_logger().warning("⚠️ No camera frame.")
            return

        if self.first_frame:
            self.camera_info = dh.get_camera_info(depth_frame)
            self.first_frame = False

        color_image = np.asanyarray(color_frame.get_data())
        self.aruco_detector.update(color_image, self.camera_info)

        all_markers = self.aruco_detector.get_detected_marker_dict()

        filtered = {
            k: v for k, v in all_markers.items() if k in self.allowed_ids
        }

        if not filtered:
            self.get_logger().info("🔍 No tracked ArUco markers found.")
            return

        for marker_id, data in filtered.items():
            pos = data['pos']
            self.get_logger().info(f"📌 Marker {marker_id} at (x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f})")


def main(args=None):
    try:
        node = MotionLoopNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
