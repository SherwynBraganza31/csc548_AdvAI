#!/usr/bin/env python3
import json
import os.path

import numpy as np
import struct
import zmq
import math

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3
LEARNING_RATE = 0.01


def roundToBins(x, bin=0.1):
    if x == 0:
        return 0
    return bin * round(x / bin)


def checkIfReset(x, xdot, theta, thetadot, reward) -> bool:
    # if out of bounds, reset
    if x < -5 or x > 5:
        return True

    # if below horizon, reset
    # if theta < -math.pi / 2 or x > math.pi / 2:
    #     return True

    # if reward == -1:
    #     return True


def roundState(state: tuple) -> tuple:
    x, xdot, theta, thetadot, reward = state
    return (roundToBins(x, 0.1),
            roundToBins(xdot, 1),
            roundToBins(theta, 0.1),
            roundToBins(thetadot, 0.1),
            reward)


def main():
    current_state = (0, 0, 0.2, 0)
    current_action = 0
    # if os.path.exists('model.json'):
    #     with open('model.json', 'r') as infile:
    #         state_action_table = json.load(infile)
    # else:
    #     state_action_table = {}
    state_action_table = {}

    reset_flag = False

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    count = 0
    animation_enabled = False
    while True:
        if count == 0:
            # toggle animation
            command = ANIMATE
            request_bytes = struct.pack('ii', command, False)
            socket.send(request_bytes)

        elif count%1000 == 0 and animation_enabled:
            # toggle animation
            command = ANIMATE
            animation_enabled = False
            request_bytes = struct.pack('ii', command, False)
            socket.send(request_bytes)

        elif count%10000 == 0 and not animation_enabled:
            # toggle animation
            command = ANIMATE
            animation_enabled = True
            request_bytes = struct.pack('ii', command, True)
            socket.send(request_bytes)

        # if count % 10000 == 0:
        #     with open('model.json', 'w') as outfile:
        #         json.dump(state_action_table, outfile)

        elif reset_flag:
            # reset the state
            command = SET_STATE
            x = 0
            xdot = 0.0
            theta = 0.2
            thetadot = 0.0
            current_state = (x, xdot, theta, thetadot)
            request_bytes = struct.pack(
                'iffff', command, x, xdot, theta, thetadot)
            socket.send(request_bytes)
            reset_flag = False

        else:
            command = APPLY_FORCE
            if current_state in state_action_table:
                state_action = state_action_table[current_state]
                best_action = max(state_action, key=state_action.get)
                if state_action[best_action] < -0.01:
                    best_action = np.round((np.random.random() * 5) - 2.5, 3)
            else:
                best_action = np.round((np.random.random() * 5) - 2.5, 3)
                Q_current = {best_action:0}
                state_action_table.update({current_state:Q_current})
            request_bytes = struct.pack('if', command, best_action)
            current_action = best_action
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])

            new_state = (x, xdot, theta, thetadot, reward)
            rounded_state = roundState(new_state)

            Q_current = state_action_table[current_state]
            current_action_Q_value = Q_current[current_action] if current_action in Q_current else 0
            best_action = max(Q_current, key=Q_current.get)
            Q_current[current_action] = current_action_Q_value \
                                        + LEARNING_RATE * \
                                        (reward + 0.95*Q_current[best_action] - current_action_Q_value)

            state_action_table.update(Q_current)
            current_state = rounded_state

            reset_flag = checkIfReset(x, xdot, theta, thetadot, reward)
            #print(current_state, best_action)

        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)


if __name__ == "__main__":
    main()
