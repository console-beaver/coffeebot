#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm 

from utils.respeaker import ReSpeaker 

class IntegratedNode(hm.HelloNode):
    def __init__(self):
        super().__init__()
        self.state = 'init'  # state machine
        self.order_received = None

        # === Init ReSpeaker helper ===
        self.respeaker = ReSpeaker(self) 
        hm.HelloNode.main(self, 'integrated_node', 'integrated_node', wait_for_first_pointcloud=False)
    
    def main(self):
        self.get_logger().info('✅ Node started.')
        self.respeaker.ask_for_order()
        self.create_timer(1.0, self.state_machine_loop)

    def state_machine_loop(self):
        if self.state == 'init':
            order = self.respeaker.check_for_order()
            if order:
                self.get_logger().info(f'📦 Got order {order}')
                self.state = 'motion'  # or transition elsewhere
        elif self.state == 'motion':
            # Add future behavior (e.g., move arm or respond)
            self.get_logger().info('🎯 Executing motion...')
