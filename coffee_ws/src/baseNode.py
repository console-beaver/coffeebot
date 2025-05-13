import hello_helpers.hello_misc as hm 
import rclpy  
from rclpy.node import Timer
import hello_helpers.hello_misc as hm
import time


class BaseNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self) 
        self.current_index = 0   

        self.lift_positions = [0.0, 1.0] 
        self.target = None 
        self.moving = False 
        self.timer = None
        
    def main(self):
        hm.HelloNode.main(self, 'my_node', 'my_node', wait_for_first_pointcloud=False)

        self.move_to_pose({'joint_lift': 0.6}, blocking=True)
        self.move_to_pose({'joint_wrist_yaw': -1.0, 'joint_wrist_pitch': -1.0}, blocking=True)
        
        self.timer = self.create_timer(0.2, self.lift_loop)

        rclpy.spin(self)  

    def lift_loop(self):
        self.robot.pull_status()
        current_lift = self.robot.status['joint_states']['position']['joint_lift'] # check this

        if not self.moving:
            self.target = self.lift_positions[self.current_index]
            self.get_logger().info(f"Moving lift to {self.target}")
            self.robot.move_to({'joint_lift': self.target})
            self.moving = True
        else:
            if abs(current_lift - self.target) < 0.01:
                self.get_logger().info(f"Reached {self.target}")
                self.current_index = 1 - self.current_index  # Toggle
                self.moving = False

if __name__ == '__main__':
    rclpy.init()
    node = BaseNode()
    node.main()