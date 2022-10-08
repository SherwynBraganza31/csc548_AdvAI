#!/usr/bin/env python3
import json
import numpy as np
import struct
import zmq
import math

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3

def myround(x, base=0.1):
    if x == 0:
        return 0
    return base * round(x/base)

def main():
    # x = [-5:0.1:5]
    # xdot = [-10:0.5:10]
    # theta = [-pi/2:pi/18:pi/2]
    # thetadot = [-pi*10:1:pi*10]
    # force = [-10:1:10]
    td_tensor = -1 * np.ones((101, 21, 37, 63, 21))
    last_action_idx = 0
    current_state = (50,10,18,18,10)
    reset_flag = False

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    animation_enabled = True
    count = 0
    while True:
        if count % 1000 == 0:
            # toggle animation
            command = ANIMATE
            animation_enabled = not animation_enabled
            request_bytes = struct.pack('ii', command, animation_enabled)
            socket.send(request_bytes)

        elif reset_flag:
            # reset the state
            command = SET_STATE
            x = np.random.randint(-1,1)
            xdot = 0.0
            theta = np.random.random(1) * (math.pi/20)
            thetadot = 0.0
            request_bytes = struct.pack(
                'iffff', command, x, xdot, theta, thetadot)
            socket.send(request_bytes)
            reset_flag = False

        else:
            command = APPLY_FORCE
            x, xdot, theta, thetadot,force = current_state
            last_action_idx = np.argmax(td_tensor[x,xdot,theta,thetadot,:])
            if not last_action_idx:
                action = -theta/math.pi * 1
            else:
                action = np.random.random(1) - 0.5
            current_state = (x, xdot, theta, thetadot, last_action_idx)
            request_bytes = struct.pack('if', command, action)
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])
            if x < -5 or x > 5:
                reset_flag = True
                continue
            if theta < -math.pi/2 or x > math.pi/2:
                reset_flag = True
                continue
            x = int(np.round(myround(x, 0.1)*10 + 50))
            xdot = int(np.round(myround(xdot, 0.5) + 10))
            theta = int(np.round(myround(theta, math.pi/180) + math.pi,0))
            thetadot = int(np.round(myround(thetadot, math.pi / 18) + math.pi, 0))
            new_state = (x, xdot, theta, thetadot, 10)
            td_tensor[current_state] += reward + td_tensor[new_state]
            current_state = new_state
            print(new_state, reward)
            if reward == -1:
                reset_flag = True


        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)


if __name__ == "__main__":
    main()
