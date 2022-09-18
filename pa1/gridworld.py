import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 5 # Grid row/col size
EIGHT_POS = [2, 0] # Position of the 8 cell
FIFTEEN_POS = [4, 4] # Position of the 15 cell
DISCOUNT = 0.95
ACTION_PROB = 0.25

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']


def step(state, action):
    """
        Edge Case Handling

        CASE 1 - Lands on 8 or Magic teleporter
        CASE 2 - Lands on 15
    """
    # if it falls on the 8 cell
    if state == EIGHT_POS and list(action) == list(ACTIONS[0]):
        return FIFTEEN_POS, -2
    # if it falls on the 15 cell
    if state == FIFTEEN_POS and (list(action) == list(ACTIONS[2]) or list(action) == list(ACTIONS[3])):
        return FIFTEEN_POS, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    # if you go off bounds
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = -1
    return next_state, reward

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]
        val = ''
        for ba in best_actions:
            val += ACTIONS_FIGS[ba]
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def stochastic_solver():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-8:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('images/pa1_valueFunc.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('images/pa1_optimal_policy.png')
            plt.close()
            break
        value = new_value

if __name__ == '__main__':
    stochastic_solver()