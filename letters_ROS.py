# USAGE: run
# `ros2 launch stretch_core stretch_driver.launch.py mode:=position`
# then, run this script while specifying a letter in {a, b, c}

# letters are defined relative to bottom-left corner, (x, y, z) (where z is out of the page)
LETTER_WAYPOINTS = {
        'a': ((0,0,0), (0.5,1,0), (1,0,0), (1,0,1), (0.75,0.5,1), (0.75,0.5,0), (0.25,0.5,0)),
        'b': ((0,0,0), (0,1,0), (1,0.75,0), (0,0.5,0), (1,0.25,0), (0,0,0)),
        'c': ((1,0,0), (0.5,0,0), (0,0.5,0), (0.5,1,0), (1,1,0))
}

import sys
import os
from stretch_control import *

if __name__ == '__main__':
    assert len(sys.argv) > 1, 'forgot letter argument'
    letter = sys.argv[1][0].lower()
    if letter in LETTER_WAYPOINTS.keys():
        import hello_helpers.hello_misc as hm

        # TODO: verify that this is the correct urdf to use (Michelle?)
        urdf_path = '/home/cs225a1/.local/lib/python3.10/site-packages/stretch_urdf/RE2V0/stretch_description_RE2V0_tool_stretch_gripper.urdf'
        chain = prep_chain(urdf_path)
        node = hm.HelloNode.quick_create('letters_demo')

        for point in LETTER_WAYPOINTS[letter]:
            # TODO: may need to add transformation to get waypoints into stretch base frame
            EE_position_control(point, node)
    else:
        print('invalid letter argument')
