import subprocess
import platform
import time
from send_req_llama import LlamaPrompt
import os 

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def run_in_new_terminal(cmd, title=None):
    """
    If $DISPLAY is available, pop up a gnome-terminal.
    Otherwise, run in the background, logging to logs/{title}.log.
    """
    if os.environ.get("DISPLAY"):
        # your existing gnome-terminal code...
        args = ["gnome-terminal"]
        if title:
            args += ["--title", title]
        args += ["--", "bash", "-c", f"{cmd}; echo 'Done. Press Enter to close'; read"]
        return subprocess.Popen(args)
    else:
        logfile = os.path.join(LOG_DIR, f"{title or 'proc'}.log")
        bg_cmd = f"{cmd} > {logfile} 2>&1"
        print(f"[no-display] launching `{bg_cmd}` → {logfile}")
        return subprocess.Popen(bg_cmd, shell=True)

def kill_process(proc):
    try:
        proc.terminate()
    except Exception:
        pass
    # optionally: proc.kill()

if __name__ == "__main__":
    # your list of object names
    list_of_objects = ["orange", "apple", "sports ball", "cup"]

    # fire off your image‐sending script once
    p_img = run_in_new_terminal("python3 send_d405_images.py", title="send_d405_images")

    # get your (object, bag) pairs from llama
    llama = LlamaPrompt(image_path="/home/cs225a1/ina/8VC-Hackathon/d405_image_10.png")
    result = llama.prompt_llama(list_of_objects)
    print("LLM result:", result)

    for object_name, bag_name in result:
        # build the two commands
        cmd1 = f"xvfb-run -a python3 recv_and_yolo_d405_images.py -c {object_name}"
        cmd2 = "python3 visual_servoing_demo.py -y -A" if bag_name == "food" else "python3 visual_servoing_demo.py -y -B"

        # launch each in its own terminal
        p1 = run_in_new_terminal(cmd1, title=f"{object_name}-yolo")
        p2 = run_in_new_terminal(cmd2, title=f"{object_name}-servo")

        print(f"→ Launched p1={p1.pid}, p2={p2.pid} for '{object_name}'")

        # **only** wait for p2
        p2.wait()
        print(f"← p2 ({p2.pid}) exited; killing p1 ({p1.pid})…")

        # now kill p1
        kill_process(p1)
        p1.wait()
        print(f"✔ Finished processing '{object_name}'\n")

    # when you're all done you can optionally kill p_img too
    kill_process(p_img)
    print("All done.")
