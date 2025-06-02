#!/usr/bin/env python3

import time
import subprocess
import sys
from utils.state_comp import barista_state
from utils.secrets import BLINKEY_IP, INKEY_IP, BLINKEY_IP2
from utils.sharedOrderQ import SharedOrderQ, order

RATE = 1.0

class BaristaNode:
    def __init__(self):
        self.state = barista_state()
        self.asked = False
        self.queue = SharedOrderQ(redis_host=BLINKEY_IP2, redis_port=6400)

        self.stations = {
            1: 'ABD',
            2: 'BCD',
            3: 'AAD',
            4: "CD"
        }

    def main(self):
        print('✅ Barista Node initialized.')
        try:
            while True:
                self.state_machine_loop()
                time.sleep(RATE)
        except KeyboardInterrupt:
            print('❗ Interrupted. Shutting down...')

    def state_machine_loop(self):
        self.queue.load_from_redis()
        if self.state.state == 'init':
            self.state.compute_state(self.queue)
        elif self.state.state == 'brewing':
            self.make_coffee()
        elif self.state.state == 'done_brewing':
            self.state.compute_state(self.queue)

    def make_coffee(self):
        coffee = self.queue.next_coffee()
        if coffee:
            print(f'☕ Making coffee for order {coffee.order_number}')
            time.sleep(2)
            for station in self.stations[coffee.order_number]:
                print(f'BARISTANODE: pouring from station {station} for order {coffee.order_number}')

                # TODO: We probably dont need to run p1 and p2 here but I would test it first 
                #print('starting first (2) processes')
                #p1 = subprocess.Popen([sys.executable, '/home/cs225a1/coffeebot/testcoffee/send_d405_images.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                #p2 = subprocess.Popen(['xvfb-run','-a','python3','/home/cs225a1/coffeebot/testcoffee/recv_and_yolo_d405_images.py','-c','cup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                #print('sleeping...')
                #time.sleep(2)

                p = subprocess.Popen([sys.executable, '/home/cs225a1/coffeebot/testcoffee/visual_servoing_demo.py', '-y', '--station', f'{station}'], stdout=subprocess.PIPE, text=True)

                for line in p.stdout:
                    if "Sequence complete. Shutting down." in line:
                        break

                p.terminate()
                p.wait()
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("Process did not terminate, killing it.")
                    p.kill()
                    p.wait()

                '''
                print('starting local process')
                p1 = None

                #TODO: Test to see if thus will kill the servovectoring after the amount of stateted time
                try:
                    p1 = subprocess.run(
                        [sys.executable, '/home/cs225a1/coffeebot/testcoffee/visual_servoing_demo.py', '-y', '--station', f'{station}'],
                        capture_output=True,
                        text=True,
                        timeout=120  #TODO: seconds, adjust as needed
                    )
                    print('p1 (visual servoing) finished')
                    if p1.stdout:
                        print(f'stdout: {p1.stdout}')
                    if p1.stderr:
                        print(f'stderr: {p1.stderr}')
                except subprocess.TimeoutExpired:
                    print('visual_servoing_demo.py timed out — forcing termination')


                #print('killing background processes')
                p1.terminate()
                #p2.terminate()
                time.sleep(1)
                if p1.poll() is None: p1.kill()
                #if p2.poll() is None: p2.kill()
                '''

            print('✅ Coffee made.')
            self.state.compute_state(self.queue)
        return

if __name__ == '__main__':
    barista = BaristaNode()
    barista.main()

"""
#!/usr/bin/env python3

import rclpy
import time
import hello_helpers.hello_misc as hm 
import subprocess
import sys

# import classes here where your command functions are 
from utils.state_comp import barista_state 
from utils.secrets import BLINKEY_IP, INKEY_IP, BLINKEY_IP2
from utils.sharedOrderQ import SharedOrderQ, order
RATE = 1.0

class BaristaNode(hm.HelloNode):
    def __init__(self): 
        super().__init__()
        self.state = barista_state() 
        self.asked = False
        self.queue = SharedOrderQ(redis_host=BLINKEY_IP2, redis_port=6400)  

        # placeholder for now, TODO: populate stations with correct list for each order
        self.stations = {
                         1: 'ABD',
                         2: 'BCD',
                         3: 'AAD', 
                         4: "CD"
                        }

        hm.HelloNode.main(self, 'barista_node', 'barista_node', wait_for_first_pointcloud=False) 

    def main(self):  
        self.get_logger().info('✅ Barista Node initialized.') 
        self.create_timer(RATE, self.state_machine_loop) 

    def state_machine_loop(self):  
        # TODO: add correct state design and include how to compute logic in utils/state_comp.py
        self.queue.load_from_redis()
        if self.state.state == 'init':  
            self.state.compute_state(self.queue)  
        elif self.state.state == 'brewing': 
            # print('The Q: ', self.queue.coffee, self.queue.labels) 
            self.make_coffee() 
            #self.state.compute_state(self.queue)
        elif self.state.state == 'done_brewing':    
            #self.make_coffee() 
            self.state.compute_state(self.queue)
    
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
                print('starting first (2) processes')
                p1 = subprocess.Popen([sys.executable, '/home/cs225a1/coffeebot/testcoffee/send_d405_images.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                p2 = subprocess.Popen(['xvfb-run','-a','python3','/home/cs225a1/coffeebot/testcoffee/recv_and_yolo_d405_images.py','-c','cup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                print('sleeping...')
                time.sleep(2)
                
                # visual servoing demo is blocking: run until complete
                print('starting local process')
                p3 = subprocess.run([sys.executable, '/home/cs225a1/coffeebot/testcoffee/visual_servoing_demo.py','-y','--station',f'{station}'], capture_output=True, text=True)
                
                print('p3 (visual servoing) finished')
                if p3.stdout:
                    print(f'stdout: {p3.stdout}')
                if p3.stderr:
                    print(f'stderr: {p3.stderr}')
                
                # now kill background processes
                print('killing background processes')
                p1.terminate()
                p2.terminate()
                time.sleep(1)
                if p1.poll() is None: p1.kill()
                if p2.poll() is None: p2.kill()

            self.get_logger().info('✅ Coffee made.') 
            self.state.compute_state(self.queue)
    
    def give_order_to_human(self): 
        # TODO: Move to completed cup, grasp, lift, and move away from location so human can grab 
        return 

# === MAIN ENTRY POINT ===
def main(args=None): 
    """ """Main entry point for the IntegratedNode. 
    This function initializes the ROS2 node and starts the main loop.
    """ """
    try: 
        node = BaristaNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        node.get_logger().info('❗ Interrupted. Shutting down...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() """
