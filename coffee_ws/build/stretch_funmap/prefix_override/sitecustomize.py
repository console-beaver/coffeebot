import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/cs225a1/coffeebot/coffee_ws/install/stretch_funmap'
