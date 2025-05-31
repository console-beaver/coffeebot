#!/usr/bin/env python3

import math
import numpy as np
import sys
sys.path.append('/home/cs225a1/coffeebot')
from letters_ROS import EE_position_control_2

INCHES_PER_METER = 39.37
EE_LENGTH = 5.45 * 2 / INCHES_PER_METER
clamp = lambda x, l, h : max(min(x, h), l)
sign = lambda x : (x > 0) - (x < 0)

"""
move_EE_to_xyz: function which implements position control on stretch
all xyz coordinates use the following coordinate frame:
    +x direction points in direction of telescoping arm extension
    +y direction points in the forward direction of the base
    +z direction points upward from the base
    -origin is on the top of the base, under the wrist's revolute joint
the function takes the following parameters:
    -X: a 3-tuple of x,y,z position in the above coordinate frame
    -node: a node object which inherits from hm.HelloNode (normally pass self)
    -sleep_time: optional # of seconds to block after movement is finished
    -blocking: optional bool, if true function blocks until movement finished
    -interp_info: optional 2-tuple of (start_position, interp_points)
        -start_position: current x,y,z position
        -interp_points: the number of interp points to add to the movement
    -interp_points: optional int, if nonzero adds points between the current
        ...position at the time of the call, and the target position (X)
    -closed: optional bool, if true closes the gripper during the movement
"""
def move_EE_to_xyz(X,  # the XYZ position
                   node,  # a node object which inherits from hm.HelloNode
                   sleep_time=0,  # time in seconds
                   blocking=True,
                   interp_info=None,  # number of points to add
                   closed=True
                  ):

    if interp_info is not None and len(interp_info) == 2 and interp_info[1] > 0:
        print('moving EE to xyz position with interp')
        start_position, interp_points = interp_info
        t_values = np.linspace(0, 1, interp_points + 2)[1:-1]
        prev_point = np.asarray(start_position, dtype=float)
        next_point = np.asarray(X, dtype=float)
        for t in t_values:
            interp_point = tuple(prev_point + t * (next_point - prev_point))
            EE_position_control_2(interp_point, node, blocking=False, sleep_time=0.15, closed=closed)
    else:
        print('moving EE to xyz position without interp')
    EE_position_control_2(X, node, sleep_time=sleep_time, blocking=blocking, closed=closed)
