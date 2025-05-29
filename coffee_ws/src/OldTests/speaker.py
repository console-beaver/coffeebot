#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import time
from stretch_body.robot import Robot

class BeepOnlyNode(Node):
    def __init__(self):
        super().__init__('beep_only_node')
        self.get_logger().info('✅ Node is ready. Beeping 3 times...')
        self.robot = Robot()
        self.robot.startup()
        self.robot.pimu.trigger_beep()
        time.sleep(0.5)
        self.robot.pimu.trigger_beep()
        time.sleep(0.5)
        self.robot.pimu.trigger_beep()
        time.sleep(0.5)
        self.get_logger().info('✅ Done beeping.')
        rclpy.shutdown()

def main():
    rclpy.init()
    node = BeepOnlyNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()

