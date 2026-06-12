#!/usr/bin/env python3
import tty, sys, termios, json, time, select

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)

def write_cmd(vx, vy, wz):
    with open('/tmp/robot_cmd.json', 'w') as f:
        json.dump({"vx": vx, "vy": vy, "wz": wz}, f)

print("Arrow keys to control robot. Q to quit.")
print("UP=forward  DOWN=backward  LEFT=turn left  RIGHT=turn right")

keys = {"up": False, "down": False, "left": False, "right": False}
last_press = {k: 0.0 for k in keys}
KEY_TIMEOUT = 0.3  # 300ms - longer than key repeat rate

try:
    tty.setraw(fd)
    while True:
        r, _, _ = select.select([sys.stdin], [], [], 0.02)  # 20ms polling

        if r:
            ch = sys.stdin.read(1)
            if ch in ('q', 'Q'):
                break
            elif ch in ('r', 'R'):
                with open('/tmp/robot_cmd.json', 'w') as f:
                    import json
                    json.dump({"vx": 0.0, "vy": 0.0, "wz": 0.0, "reset": True}, f)
                time.sleep(1.0)  # hold reset flag long enough for sim to read
            elif ch == '\x1b':
                rest = sys.stdin.read(2)
                t = time.time()
                if rest == '[A':   keys["up"] = True;    last_press["up"] = t
                elif rest == '[B': keys["down"] = True;  last_press["down"] = t
                elif rest == '[D': keys["left"] = True;  last_press["left"] = t
                elif rest == '[C': keys["right"] = True; last_press["right"] = t

        # release keys that haven't been pressed recently
        now = time.time()
        for k in keys:
            if keys[k] and now - last_press[k] > KEY_TIMEOUT:
                keys[k] = False

        vx = 0.0
        wz = 0.0
        if keys["up"]:    vx += 1.0
        if keys["down"]:  vx -= 0.5
        if keys["left"]:  wz += 3.0
        if keys["right"]: wz -= 3.0
        if vx == 0.0 and wz == 0.0: vx = 0.3  # slow default

        write_cmd(vx, 0.0, wz)

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    write_cmd(0, 0, 0)
# Note: R key already handled - add to controller
