#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm 
import subprocess
import sys

# import classes here where your command functions are 
from utils.state_comp import orderQ, order, barista_state 

RATE = 1.0

class BaristaNode(hm.HelloNode):
    def __init__(self): 
        super().__init__()
        self.state = barista_state() 
        self.asked = False
        self.queue = orderQ() #TODO: both robots have to share the same queue object  

        self.stations = {
                         1: 'AB',
                         2: 'BC',
                         3: 'A'
                        }

        hm.HelloNode.main(self, 'barista_node', 'barista_node', wait_for_first_pointcloud=False) 

    def main(self):  
        self.get_logger().info('✅ Barista Node initialized.') 
        self.create_timer(RATE, self.state_machine_loop) 

    def state_machine_loop(self):  
        # TODO: add correct state design and include how to compute logic in utils/state_comp.py
        if self.state.state == 'init': 
            self.state.compute_state(self.queue) 
        elif self.state.state == 'making coffee 1':    
            self.make_coffee()
        elif self.state.state == 'making coffee 2': 
            self.make_coffee()
        elif self.state.state == 'making coffee 3':
            self.make_coffee()
        elif self.state.state == 'pouring': 
            self.make_coffee() 
    
    def make_coffee(self):  
        # TODO: inclucde logic preferably we make commads in a seperate file and call them here 
        # We could do function from another file reach_for_ingredient(), grab(), move(x), ect...
        coffee = self.queue.next_coffee() 
        if coffee:
            self.get_logger().info(f'☕ Making coffee for order {coffee.order_number}') 
            time.sleep(2)
            for station in self.stations[coffee.order_number]:
                print(f'BARISTANODE: pouring from station {station} for order {coffee.order_number}')
                # launch necessary background processes
                p1 = subprocess.Popen([sys.executable, 'python3 send_d405_images.py'])
                p2 = subprocess.Popen([sys.executable, 'xvfb-run -a python3 recv_and_yolo_d405_images.py -c cup'])

                # visual servoing demo is blocking: run until complete
                p3 = subprocess.run([sys.executable, f'python3 visual_servoing_demo.py -y --station {station}'], capture_output=True)
                
                # now kill background processes
                p1.terminate()
                p2.terminate()
                time.sleep(1)
                if p1.poll() is None: p1.kill()
                if p2.poll() is None: p2.kill()
            self.get_logger().info('✅ Coffee made.')  
    
    def give_order_to_human(self): 
        # TODO: Move to completed cup, grasp, lift, and move away from location so human can grab 
        return 

# === MAIN ENTRY POINT ===
def main(args=None): 
    """Main entry point for the IntegratedNode. 
    This function initializes the ROS2 node and starts the main loop.
    """
    try: 
        node = BaristaNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
