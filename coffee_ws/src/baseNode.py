#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm 

class MotionLoopNode(hm.HelloNode):
    """
    MotionLoopNode: serves as a base node to help control the motion commands over time
    """
    def __init__(self):
        super().__init__()
        self.rate = 0.5
        self.total_time = 60
        self.state = 0  # 0 = Pose A, 1 = Pose B
        self.poses = [
            {'joint_lift': 0.4, 'joint_wrist_yaw': -1.0, 'joint_wrist_pitch': -1.0},  # Pose A
            {'joint_lift': 1.0, 'joint_wrist_yaw': 0.0, 'joint_wrist_pitch': 0.0},    # Home
        ]
        self.start_time = None
        self.motion_timer = None

        # 🧠 Start the HelloNode system (including spin thread)
        hm.HelloNode.main(self, 'motion_loop_node', 'motion_loop_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node is ready. Switching to position mode...')
        self.switch_to_position_mode()
        time.sleep(0.5)

        self.start_time = time.time()
        self.get_logger().info(f'Starting {self.rate}-second motion timer.')
        self.motion_timer = self.create_timer(self.rate, self.motion_loop)
    def motion_loop(self):
        elapsed = time.time() - self.start_time
        if elapsed > self.total_time:
            self.get_logger().info(f"✅ {self.total_time} sec complete. Shutting down.")
            self.motion_timer.cancel()
            rclpy.shutdown()
            return

        pose = self.poses[self.state]
        self.get_logger().info(f'🤖 Moving to pose: {pose}')
        try:
            self.move_to_pose(pose, blocking=True)
        except Exception as e:
            self.get_logger().error(f'❌ Motion failed: {e}')
            rclpy.shutdown()

        self.state = 1 - self.state

def main(args=None):
    try:
        node = MotionLoopNode()
        node.main()
        node.new_thread.join()  # Required to block until node finishes
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupt received. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
# Example documentation from Hello RObot tutorial
import hello_helpers.hello_misc as hm

class MyNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)

    def main(self):
        hm.HelloNode.main(self, 'my_node', 'my_node', wait_for_first_pointcloud=False)

        # my_node's main logic goes here
        self.move_to_pose({'joint_lift': 1.0}, blocking=True)
        self.move_to_pose({'joint_wrist_yaw': -1.0, 'joint_wrist_pitch': -1.0}, blocking=True)

node = MyNode()
node.main()
"""