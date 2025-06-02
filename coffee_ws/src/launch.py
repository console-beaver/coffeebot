import subprocess
import time

# Launch Redis: delete key and start server
print('Preparing to clear and run redis-server...')
redis_cmd = """
redis-cli -h 127.0.0.1 -p 6379 DEL shared_orderQ
redis-server --port 6400 --protected-mode no
"""
redis_server = subprocess.Popen(redis_cmd, shell=True, executable="/bin/bash")
time.sleep(5)

# Launch Stretch Core
print('Launching stretch core ...')
stretch = subprocess.Popen(["ros2", "launch", "stretch_core", "stretch_driver.launch.py"])
time.sleep(5)

# Launch ReSpeaker
print('Launching respeaker ...')
respeaker = subprocess.Popen(["ros2", "launch", "respeaker_ros2", "respeaker.launch.py"])

# Wait for everything
try:
    stretch.wait()
    respeaker.wait()
except KeyboardInterrupt:
    stretch.terminate()
    respeaker.terminate()
    redis_server.terminate()
