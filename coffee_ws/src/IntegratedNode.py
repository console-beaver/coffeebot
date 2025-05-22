#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm
from utils.respeaker import ReSpeaker  # 🔁 Your helper class

class IntegratedNode(hm.HelloNode):
    def __init__(self):
        super().__init__()
        self.state = 'init'
        self.respeaker = ReSpeaker(self)

        hm.HelloNode.main(self, 'integrated_node', 'integrated_node', wait_for_first_pointcloud=False)

    def main(self):
        self.get_logger().info('✅ Node initialized. Asking for order.')
        self.respeaker.ask_for_order()

        self.create_timer(1.0, self.state_machine_loop)

    def state_machine_loop(self):
        if self.state == 'init':
            order = self.respeaker.check_for_order()
            if order:
                self.get_logger().info(f'📦 Order received: {order}')
                self.state = 'done'
        elif self.state == 'done':
            self.get_logger().info('✅ Task complete.')
            rclpy.shutdown()

# === MAIN ENTRY POINT ===
def main(args=None):
    try:
        node = IntegratedNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
