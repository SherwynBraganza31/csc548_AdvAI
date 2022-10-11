#!/usr/bin/env python3
import json
import os

import numpy as np
import struct
import zmq

APPLY_FORCE = 0
SET_STATE = 1
NEW_STATE = 2
ANIMATE = 3
LEARNING_RATE = 0.01

def roundToBins(x, bin=0.1):
    """
        Author: Sherwyn Braganza

        Rounds the number x to the specified bins. For example, if we want to round the number on a 0.5 scale,
        1.8 would be rounded to 2 and 0.7 would be rounded to 0.5.

        :param x: The number to be rounded
        :param bin: The bound along which the number should be rounded
        :return: The rounded number
    """
    if x == 0:
        return 0
    return bin * round(x / bin)


def checkIfReset(x, xdot, theta, thetadot, reward) -> bool:
    """
        Checks the individual state params and sets the reset Flag if
        either one of them meets the reset specifications

        :param x: State param corresponding to position
        :param xdot: State param corresponding to velocity
        :param theta: State param corresponding angle
        :param thetadot: State param corresponding to angular velocity
        :param reward: The reward from taking the action and being in state s+1
        :return: True if need to reset, False otherwise
    """
    # if out of bounds, reset
    if x < -5 or x > 5:
        return True

    # if below horizon, reset
    # if theta < -math.pi / 2 or x > math.pi / 2:
    #     return True

    # if reward == -1:
    #     return True

    return False


def roundState(state: tuple) -> tuple:
    """
        Author: Sherwyn Braganza

        Rounds the state params to their respective bins

        :param state: The state params to be rounded
        :return: the rounded state params
    """
    x, xdot, theta, thetadot, reward = state
    return (roundToBins(x, 0.1),
            roundToBins(xdot, 1),
            roundToBins(theta, 0.1),
            roundToBins(thetadot, 0.1),
            reward)


def executeSARSA(current_state, state_action_table):
    # check if the state is already experienced and select the best Action
    # Follows the SARSA method (On-Policy aka the best possible action at each state)
    if current_state in state_action_table:
        state_action = state_action_table[current_state]
        best_action = max(state_action, key=state_action.get)

        # if state has a large negative value, randomize the force value than choosing the best action
        if state_action[best_action] < -2:
            best_action = np.round((np.random.random() * 5) - 2.5, 3)
    else:
        # if states not experienced, randomize an action value and
        best_action = np.round((np.random.random() * 5) - 2.5, 3)

        # store the State - Action Pair with value 0
        Q_current = {best_action: 0}
        state_action_table.update({current_state: Q_current})

    return best_action


def updateStateActionTable(current_state, state_action_table, current_action, rounded_state):
    Q_current = state_action_table[current_state]
    current_action_Q_value = Q_current[current_action] if current_action in Q_current else 0
    best_action = max(Q_current, key=Q_current.get)
    Q_current[current_action] = current_action_Q_value \
                                + LEARNING_RATE * \
                                (rounded_state[4] + 0.95 * Q_current[best_action] - current_action_Q_value)

    state_action_table.update(Q_current)

    return rounded_state[0], rounded_state[1], rounded_state[2], rounded_state[3]

def main():
    # state action variables
    current_state = (0, 0, 0.2, 0)
    current_action = 0

    # if os.path.exists('model.json'):
    #     with open('model.json', 'r') as infile:
    #         state_action_table = json.load(infile)
    # else:
    #     state_action_table = {}
    state_action_table = {}

    reset_flag = False

    # ZMQ Socket Setup
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

        # elif count%1000 == 0 and animation_enabled:
        #     # toggle animation
        #     command = ANIMATE
        #     animation_enabled = False
        #     request_bytes = struct.pack('ii', command, False)
        #     socket.send(request_bytes)

        elif count%1000 == 0 and not animation_enabled:
            # toggle animation
            command = ANIMATE
            animation_enabled = True
            request_bytes = struct.pack('ii', command, True)
            socket.send(request_bytes)

        # if count % 10000 == 0:
        #     with open('model.json', 'w') as outfile:
        #         json.dump(state_action_table, outfile)

        elif reset_flag:
            # reset the state if the reset flag is on
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
            best_action = executeSARSA(current_state, state_action_table)
            request_bytes = struct.pack('if', command, best_action)
            current_action = best_action
            socket.send(request_bytes)

        count += 1

        response_bytes = socket.recv()
        response_command, = struct.unpack('i', response_bytes[0:4])

        # Decode the response and process
        if response_command == NEW_STATE:
            x, xdot, theta, thetadot, reward = struct.unpack(
                'fffff', response_bytes[4:])

            new_state = (x, xdot, theta, thetadot, reward)
            rounded_state = roundState(new_state)

            # update state action table
            current_state = updateStateActionTable(current_state, state_action_table, current_action, rounded_state)
            reset_flag = checkIfReset(x, xdot, theta, thetadot, reward)

        elif response_command == ANIMATE:
            animation_enabled, = struct.unpack('i', response_bytes[4:])
        else:
            print("Error: invalid command: ", response_command)


if __name__ == "__main__":
    main()
