#!/usr/bin/env python3
# USAGE: run
# `ros2 launch stretch_core stretch_driver.launch.py mode:=position`
# then, run this script while specifying a letter in {a, b, c}

# letters are defined relative to bottom-left corner, (x, y, z) (where z is out of the page)
LETTER_WAYPOINTS = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0)),
        # 'coord_test': ((0,0,0), (1,0,0), (0,0,0), (0,1,0), (0,0,0), (0,0,1), (0,0,0))
        'coord_test': ((0,0,0), (1,0,0), (0,0,0), (0,1,0), (0,0,0), (0,0,1), (0,0,0))
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

    def main(self, chain, waypoints):
        self.get_logger().info('✅ Node is ready. Switching to position mode...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        self.waypoints = waypoints
        self.chain = chain
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

        # pose = self.poses[self.state]
        self.get_logger().info(f'🤖 Moving to pose')
        s = 0.4  # scale factor for letter
        simple_transform = lambda x : (-s*x[0], -s*x[1], -s*x[2] + 0.7)
        # simple_transform = lambda x : (s * x[0] + 0.5, s * x[1], s * x[2] + 1.5)  # draw letter 0.5 meters away from base, 1 meter above ground
        # Y is pointing up, so swap Y and Z in the transform
        point = simple_transform(self.waypoints[self.waypoint_idx])
        try:
            print(f'MOVING IDX={self.waypoint_idx}, WAYPOINT={point}')
            EE_position_control(point, self, self.chain)
            # self.move_to_pose(pose, blocking=True)
        except Exception as e:
            self.get_logger().error(f'❌ Motion failed: {e}')
            rclpy.shutdown()

        self.waypoint_idx += 1

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'forgot letter argument'
    letter = sys.argv[1].lower()
    print('LETTERS: starting now')
    if letter in LETTER_WAYPOINTS.keys():
        node = LettersNode()
        print('finished creating node')
        try:
            # TODO: verify that this is the correct urdf to use (Michelle?)
            urdf_path = '/home/cs225a1/.local/lib/python3.10/site-packages/stretch_urdf/RE2V0/stretch_description_RE2V0_tool_stretch_gripper.urdf'
            chain = prep_chain(urdf_path)
            print('finished creating chain')

            print('LETTERS: created chain')
            node.main(chain, LETTER_WAYPOINTS[letter])
            node.new_thread.join()  # block until finish
        except:
            node.get_logger().info('interrupt received, shutting down')
            node.destroy_node()
            rclpy.shutdown()
    else:
        print('invalid letter argument')
