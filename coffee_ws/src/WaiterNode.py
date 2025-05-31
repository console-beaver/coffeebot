#!/usr/bin/env python3

# LAUNCHING THIS NODE REQUIRES THE FOLLOWING DRIVERS:
# ros2 launch stretch_core stretch_driver.launch.py
# ros2 launch respeaker_ros2 respeaker.launch.py

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
EE_LENGTH = 0.14

LETTER_HEIGHT = 0.04
LETTER_WIDTH = -0.03

LETTER_HEIGHT = 0.025
LETTER_WIDTH = -0.015

class WaiterNode(hm.HelloNode):
    def __init__(self): 
        super().__init__()
        self.state = waiter_state() 
        self.asked = False
        self.queue = SharedOrderQ(redis_host='localhost', redis_port=6400, redis_key='shared_orderQ') 
        
#       all xyz coordinates use the following coordinate frame:
#       +x direction points in direction of telescoping arm extension
#       +y direction points in the forward direction of the base
#       +z direction points upward from the base
        self.waypoints = {
                          # 'home': ((0.0, 0.0, MARKER_UP), (0.0, 0.02, MARKER_UP), (0.0, 0.05, MARKER_UP), (0.0, 0.07, MARKER_UP), (0.0, 0.10, MARKER_UP)),
                          'xtest': [],
                          'ztest': [],
                          'ytest': [],
                          # draw E
                          1: ( (0,LETTER_WIDTH,MARKER_UP), (0,LETTER_WIDTH,MARKER_DOWN), (0,0,MARKER_DOWN),
                              (LETTER_HEIGHT,0,MARKER_DOWN), (LETTER_HEIGHT,LETTER_WIDTH,MARKER_DOWN), 
                              (LETTER_HEIGHT,LETTER_WIDTH,MARKER_UP), (LETTER_HEIGHT/2,0,MARKER_UP), (LETTER_HEIGHT/2,0,MARKER_DOWN),
                              (LETTER_HEIGHT/2,LETTER_WIDTH,MARKER_DOWN), (LETTER_HEIGHT/2,LETTER_WIDTH,MARKER_UP), (0,0,MARKER_UP) ),
                          # draw A
                          3: ( (0,0,MARKER_UP), (0,0,MARKER_DOWN), (LETTER_HEIGHT,LETTER_WIDTH/2,MARKER_DOWN), (0,LETTER_WIDTH,MARKER_DOWN),
                              (0,LETTER_WIDTH,MARKER_UP), (LETTER_HEIGHT/2,LETTER_WIDTH/4,MARKER_UP), (LETTER_HEIGHT/2,LETTER_WIDTH/4,MARKER_DOWN),
                              (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_DOWN), (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_UP) ),
                          # draw R
                          2: ( (0,0,MARKER_UP), (0,0,MARKER_DOWN), (LETTER_HEIGHT,0,MARKER_DOWN), (LETTER_HEIGHT,LETTER_WIDTH*3/4,MARKER_DOWN),
                              (LETTER_HEIGHT*7/8,LETTER_WIDTH,MARKER_DOWN), (LETTER_HEIGHT*5/8,LETTER_WIDTH,MARKER_DOWN),
                              (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_DOWN), (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_UP),
                              (LETTER_HEIGHT/2,0,MARKER_UP),(LETTER_HEIGHT/2,0,MARKER_DOWN), (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_DOWN),
                              (LETTER_HEIGHT/2,LETTER_WIDTH*3/4,MARKER_UP), (LETTER_HEIGHT/2,0,MARKER_UP),
                              (LETTER_HEIGHT/2,0,MARKER_DOWN), (0,LETTER_WIDTH,MARKER_DOWN), (0,LETTER_WIDTH,MARKER_UP) ),
                          # draw O
                          4: ( (0,LETTER_WIDTH/4,MARKER_UP), (0,LETTER_WIDTH/4,MARKER_DOWN), (LETTER_HEIGHT/4,0,MARKER_DOWN),
                              (LETTER_HEIGHT*3/4,0,MARKER_DOWN), (LETTER_HEIGHT,LETTER_WIDTH/4,MARKER_DOWN), (LETTER_HEIGHT,LETTER_WIDTH*3/4,MARKER_DOWN),
                              (LETTER_HEIGHT*3/4,LETTER_WIDTH,MARKER_DOWN), (LETTER_HEIGHT/4,LETTER_WIDTH,MARKER_DOWN), (0,LETTER_WIDTH*3/4,MARKER_DOWN),
                              (0,LETTER_WIDTH/4,MARKER_DOWN), (0,LETTER_WIDTH/4,MARKER_UP) )
                         }

        for i in range(7):
            self.waypoints['xtest'].append((EE_LENGTH + 0.05 * i, 0.0, MARKER_UP))
            self.waypoints['ztest'].append((EE_LENGTH + 0.1, 0.0, MARKER_UP + 0.5 * i))
            self.waypoints['ytest'].append((EE_LENGTH + 0.05, -0.1 + 0.03 * i, MARKER_UP))

        hm.HelloNode.main(self, 'waiter_node', 'waiter_node', wait_for_first_pointcloud=False) 



    def main(self):  
        print('HELLO')
        self.respeaker = ReSpeaker(self) # self is the node itself 
        self.get_logger().info('✅ Waiter Node initialized.') 
        self.create_timer(RATE, self.state_machine_loop) 

        print('DEBUG')

        # self.write_label()

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
            self.get_logger().info(f'📦 Order received: {order_id}')   
            order_obj = order(order_id) 
            self.queue.add_order(order_obj) 
            order_id = None
            self.get_logger().info('✅ Ording Task complete.')   
            self.asked = False
            self.state.compute_state(self.queue) 
            print('the Q:', self.queue.labels, self.queue.coffee)
    def write_label(self):  
        """" TODO: add movement logic commands in order to write the label """ 
        order_number = self.queue.next_label().order_number
        self.get_logger().info(f'🖊️ Writing label for order {order_number}') 
        if order_number is None: return
        time.sleep(2)  
        #self.queue.next_label()
        letter_transform = lambda X: (X[0]+EE_LENGTH,X[1]+0.05,X[2])
        last_point = None
        for point in self.waypoints[order_number]:
            if order_number in (1,2,3,4):
                if not last_point is None: move_EE_to_xyz(letter_transform(point), self, sleep_time=0.25, interp_info=(last_point, 10))
                else: move_EE_to_xyz(letter_transform(point), self, sleep_time=2)
            else: move_EE_to_xyz(point, self, sleep_time=2)
            last_point = letter_transform(point)

        """for i, order_num in enumerate([1,2,3,4]):
            letter_transform = lambda X: (X[0]+EE_LENGTH + 0.035 * i ,X[1]+0.10,X[2])
            last_point = None
            for point in self.waypoints[order_num]:
                if order_num in (1,2,3,4):
                    if not last_point is None: move_EE_to_xyz(letter_transform(point), self, sleep_time=0.25, interp_info=(last_point, 10))
                    else: move_EE_to_xyz(letter_transform(point), self, sleep_time=2)
                else: move_EE_to_xyz(point, self, sleep_time=2)
                last_point = letter_transform(point)"""

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
