#!/usr/bin/env python3
# USAGE: run
# `ros2 launch stretch_core stretch_driver.launch.py mode:=position`
# then, run this script while specifying a letter in {a, b, c}

WAIT_TIME = 1
INTERP_POINTS = 10

# letters are defined relative to bottom-left corner, (x, y, z) (where z is out of the page)
LETTER_WAYPOINTS = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0)),
        'coord_test': ((0,0,0), (1,0,0), (0,0,0), (0,1,0), (0,0,0), (0,0,1), (0,0,0)),
        'yaw_test_1': ((0,0,0), (0,1,0), (0,0.5,0), (0,0,0), (0,-0.5,0), (0,-1,0)),
        'yaw_test_2': ((0,0,0), (0,0.1,0), (0,0.2,0), (0,0.3,0), (0,0.4,0), (0,0.5,0), (0,0.6,0), (0,0.7,0), (0,0.8,0), (0,0.9,0), (0,1,0)),
        'interp_test': ((0,0,0), (0,1,0))
}

import rclpy
import time
import hello_helpers.hello_misc as hm
import sys
import os
from stretch_control import *

class LettersNode(hm.HelloNode):
    """
    MotionLoopNode: serves as a base node to help control the motion commands over time
    """
    def __init__(self):
        super().__init__()
        self.rate = 0.5
        self.total_time = 60
        self.start_time = None
        self.motion_timer = None
        self.waypoint_idx = None

        # 🧠 Start the HelloNode system (including spin thread)
        hm.HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self, waypoints):
        self.get_logger().info('✅ Node is ready. Switching to position mode...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        self.waypoints = waypoints
        self.waypoint_idx = 0

        self.start_time = time.time()
        self.get_logger().info(f'Starting {self.rate}-second motion timer.')
        self.motion_timer = self.create_timer(self.rate, self.motion_loop)
    def motion_loop(self):
        if self.waypoint_idx is None:
            self.get_logger().info(f'waypoint info is not defined yet')
            return
        if self.waypoint_idx >= len(self.waypoints):
            self.get_logger().info(f'finished going through all waypoints')
            return

        elapsed = time.time() - self.start_time
        if elapsed > self.total_time:
            self.get_logger().info(f"✅ {self.total_time} sec complete. Shutting down.")
            self.motion_timer.cancel()
            rclpy.shutdown()
            return

        self.get_logger().info(f'🤖 Moving to pose')
        s = 0.2  # scale factor for letter
        simple_transform = lambda x : (-s*x[0]+0.4, -s*x[1], s*x[2] + 0.7)
        # simple_transform = lambda x : (s * x[0] + 0.5, s * x[1], s * x[2] + 1.5)  # draw letter 0.5 meters away from base, 1 meter above ground
        # Y is pointing up, so swap Y and Z in the transform
        point = simple_transform(self.waypoints[self.waypoint_idx])
        try:
            if self.waypoint_idx > 0 and INTERP_POINTS > 0:
                t_values = np.linspace(0, 1, INTERP_POINTS + 2)[1:-1]
                prev_point = np.asarray(simple_transform(self.waypoints[self.waypoint_idx-1]), dtype=float)
                next_point = np.asarray(point, dtype=float)
                for t in t_values:
                    interp_point = tuple(prev_point + t * (next_point - prev_point))
                    EE_position_control_2(interp_point, self, blocking=False, sleep_time=0.15)
            EE_position_control_2(point, self, sleep_time=5 if self.waypoint_idx == 0 else WAIT_TIME)
        except Exception as e:
            self.get_logger().error(f'❌ Motion failed: {e}')
            rclpy.shutdown()

        self.waypoint_idx += 1

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'forgot letter argument'
    letter = sys.argv[1].lower()
    if letter in LETTER_WAYPOINTS.keys():
        node = LettersNode()
        try:
            node.main(LETTER_WAYPOINTS[letter])
            node.new_thread.join()  # block until finish
        except:
            node.get_logger().info('interrupt received, shutting down')
            node.destroy_node()
            rclpy.shutdown()
    else:
        print('invalid letter argument')
