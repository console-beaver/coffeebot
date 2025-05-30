#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm  

# import classes here where your command functions are 
from utils.respeaker import ReSpeaker 
from utils.state_comp import waiter_state 
from utils.sharedOrderQ import SharedOrderQ, order  
from utils.ee_pos_control import move_EE_to_xyz

RATE = 1.0
MARKER_UP = 1.0
MARKER_DOWN = 1 - 0.022

class WaiterNode(hm.HelloNode):
    def __init__(self): 
        super().__init__()
        self.state = waiter_state() 
        self.asked = False
        self.queue = SharedOrderQ(redis_host='localhost') 

#       all xyz coordinates use the following coordinate frame:
#       +x direction points in direction of telescoping arm extension
#       +y direction points in the forward direction of the base
#       +z direction points upward from the base
        self.waypoints = {
                          # 'home': ((0.0, 0.0, MARKER_UP), (0.0, 0.02, MARKER_UP), (0.0, 0.05, MARKER_UP), (0.0, 0.07, MARKER_UP), (0.0, 0.10, MARKER_UP)),
                          'home': ((0.0, 0.0, MARKER_UP)),
                          '1': ()
                         }

        hm.HelloNode.main(self, 'waiter_node', 'waiter_node', wait_for_first_pointcloud=False) 



    def main(self):  
        print('HELLO')
        self.respeaker = ReSpeaker(self) # self is the node itself 
        self.get_logger().info('✅ Waiter Node initialized.') 
        self.create_timer(RATE, self.state_machine_loop) 

        print('DEBUG')

        self.write_label()

    def state_machine_loop(self): 
        if self.state.state == 'init': 
            self.state.compute_state(self.queue) 
        elif self.state.state == 'collecting_order':    
            self.take_order()
        elif self.state.state == 'writing_label':
            self.write_label()
    
    def take_order(self):    
        if not self.asked: 
            self.get_logger().info('Asking for order: ')
            self.respeaker.ask_for_order() 
            self.asked = True
        order_id = self.respeaker.check_for_order()
        if order_id:
            self.get_logger().info(f'📦 Order received: {order}')   
            order_obj = order(order_id)
            self.queue.add_order(order_obj)   
            self.get_logger().info('✅ Ording Task complete.')   
            self.asked = False
            self.state.compute_state(self.queue)
    def write_label(self):  
        """" TODO: add movement logic commands in order to write the label """ 
        self.get_logger().info(f'🖊️ Writing label for order {self.queue.next_label()}') 
        time.sleep(2)  
        #self.queue.next_label()
        #for point in self.waypoints['home']:
           # move_EE_to_xyz(point, self, sleep_time=2)
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
