import subprocess
import sys
import time

station = 'A'

# launch necessary background processes
print('starting first processes')
p1 = subprocess.Popen([sys.executable, 'send_d405_images.py'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
p2 = subprocess.Popen(['xvfb-run','-a','python3','recv_and_yolo_d405_images.py','-c','cup'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print('sleeping...')
time.sleep(5)

# visual servoing demo is blocking: run until complete
print('starting local process')
p3 = subprocess.run([sys.executable, '/home/cs225a1/coffeebot/testcoffee/visual_servoing_demo.py','-y','--station',f'{station}'])

print('p3 finished')
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
