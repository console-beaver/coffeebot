#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm 

# import classes here where your command functions are 
from utils.respeaker import ReSpeaker 
from utils.state_comp import orderQ, order, waiter_state, barista_state 

RATE = 1.0

class WaiterNode(hm.HelloNode):
    def __init__(self): 
        super().__init__()
        self.state = waiter_state() 
        self.asked = False
        self.queue = orderQ() #TODO: both robots have to share the same queue object  

        hm.HelloNode.main(self, 'waiter_node', 'waiter_node', wait_for_first_pointcloud=False) 

    def main(self):  
        self.respeaker = ReSpeaker(self) # self is the node itself 
        self.get_logger().info('✅ Waiter Node initialized.') 
        self.create_timer(RATE, self.state_machine_loop) 

    def state_machine_loop(self): 
        print(self.state.state)
        if self.state.state == 'init': 
            self.state.compute_state(self.queue) 
        elif self.state.state == 'collecting_order':  
            #self.asked = False  
            self.take_order()
        elif self.state.sate == 'writing_label':
            self.write_label
    
    def take_order(self):   
        print('running take order')
        if not self.asked:  
            self.respeaker.ask_for_order() 
            self.asked = True
        order_id = self.respeaker.check_for_order()
        if order_id:
            self.get_logger().info(f'📦 Order received: {order}')
            #self.state = 'done'   
            order_obj = order(order_id)
            self.queue.add_order(order_obj)   
            self.get_logger().info('✅ Ording Task complete.') 
            self.state.compute_state(self.queue)  
    def write_label(self): 
        self.get_logger().info(f'🖊️ Writing label for order {self.queue.next_label()}') 
        time.sleep(2)
        self.get_logger().info('✅ Label written.') 
        self.state.compute_state(self.queue)  
        
# === MAIN ENTRY POINT ===
def main(args=None): 
    """Main entry point for the IntegratedNode. 
    This function initializes the ROS2 node and starts the main loop.
    """
    try: 
        node = WaiterNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
