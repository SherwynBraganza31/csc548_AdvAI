'''

    CSC 548 Programming Assignment 1 - Find policies for the given grid.
    Author: Sherwyn Braganza

    Credit to Shangtong Zhang(zhangshangtong.cpp@gmail.com) and Kenta Shimada(hyperkentakun@gmail.com) for draw_image
    and draw_policy functions

    15 Sep 2022 - Initial Creation - Sherwyn Braganza
    16 Sep 2022 - Added Stochastic Solver - Sherwyn Braganza
    17 Sep 2022 - Added Deterministic Solver - Sherwyn Braganza
    18 Sep 2022 - Finalized Stochastic Solver - Sherwyn Braganza
    18 Sep 2022 - Finalized Deterministic Solver - Sherwyn Braganza
    19 Sep 2022 - Added more comments - Sherwyn Braganza

    Assignment Document/Details @ https://github.com/SherwynBraganza31/csc548_AdvAI/blob/main/pa1/pa1.pdf


    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15

    The dynamics of this variation of the Gridworld are described as follows:
    • States 8 and 15 are special.
    – State 8 is a “magic teleporter”, but using it is expensive. In state 8, the LEFT action takes you
    to state 15 with probability 1 and immediate reward -2.
    – In state 15, the RIGHT and DOWN actions take you back to state 15 with probability 1 and
    immediate reward 0.
    • In all states other than states 8 and 15, choosing an action that would move you off the grid will move
    you back into the original state with a reward of -1.
    • In all remaining cases, choosing an action will move you to an adjacent state within the grid with
    probability 1 and an immediate reward of -1. For example, choosing the DOWN action in state 5 will
    take you to state 9 with probability 1 and immediate reward -1

    Running instructions: python3 gridworld.py

    * The plots generated are stored with their resp filenames.
'''


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 4 # Grid row/col size
EIGHT_POS = [2, 0] # Position of the 8 cell
FIFTEEN_POS = [3, 3] # Position of the 15 cell
DISCOUNT = 0.95
ACTION_PROB = 0.25

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']


def step(state, action):
    '''
        Returns the Q value function for the state-action pair according to the rules

        @params:    state:list = current position in the grid
                    action:list = action to be taken based on ACTIONS

        @returns    next_state:list = the new state position
                    reward:float = the immediate reward for transitioning into said state.
    '''

    # Edge Case Handling
    # CASE 1 - Lands on 8 or Magic teleporter
    # CASE 2 - Lands on 15
    # if it falls on the 8 cell
    if state == EIGHT_POS and list(action) == list(ACTIONS[0]):
        return FIFTEEN_POS, -2
    # if it falls on the 15 cell
    if state == FIFTEEN_POS and (list(action) == list(ACTIONS[2]) or list(action) == list(ACTIONS[3])):
        return FIFTEEN_POS, 0

    next_state = (np.array(state) + action).tolist()
    x, y = next_state

    # Boundary checker
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = -1.0
    return next_state, reward


def draw_image(image):
    '''
        Takes in the value function map generated and draws it.

        Author: Sherwyn Braganza

        Credit to Shangtong Zhang(zhangshangtong.cpp@gmail.com) and Kenta Shimada(hyperkentakun@gmail.com) for
        base code

        @params     image:np.matrix : 2D matrix value function map

        @returns    none
    '''
    fig, ax = plt.subplots()
    ax.set_axis_off()
    # plt.title('Optimal Value function after {} iterations'.format(k))
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # # empty whitespace so title doesn't appear on top of col_labels
    # for i in range(len(image)):
    #     tb.add_cell(-1, i, width, height / 2, text='', loc='right',
    #                 edgecolor='none', facecolor='none')

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
    '''
        Takes in the policy map generated and draws it.

        Author: Sherwyn Braganza

        Credit to Shangtong Zhang(zhangshangtong.cpp@gmail.com) and Kenta Shimada(hyperkentakun@gmail.com) for
        base code

        @params     optimal_values:np.matrix : 2D matrix optimal action map

        @returns    none
    '''
    fig, ax = plt.subplots()
    ax.set_axis_off()
    #plt.title('Optimal Policy after {} iterations'.format(k))
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # # empty whitespace so title doesn't appear on top of col_labels
    # for i in range(len(optimal_values)):
    #     tb.add_cell(-1, i, width, height / 2, text='', loc='right',
    #                 edgecolor='none', facecolor='none')

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
    '''
        Iterative Solver that goes through the grid multiple times, improving the estimate of the value function
        with each iteration.

        Author: Sherwyn Braganza

        @params     none

        @returns    none
    '''
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    k = 0   # the k'th iteration in a stochiastic solver
    while True:
        k += 1 # keep track of the number of iterations
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action) # get imm reward and state_trans for action
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)

        # print figures if the Value Functions between two iterations is less than tol
        if np.sum(np.abs(new_value - value)) < 1e-8:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('pa1_valueFunc_stochastic.png')
            # plt.show()
            plt.close()
            draw_policy(new_value)
            plt.savefig('pa1_optimal_policy_stochastic.png')
            # plt.show()
            plt.close()

        # if no other stochastic policies can be found, break
        if np.sum(np.abs(new_value - value)) == 0 and np.array_equal(new_value, value):
            print('No other optimal stochastic policies were found, stopped at the {}\'th iteration\n'.format(k))
            break

        value = new_value

    print('The max number of optimal policies (deterministic and stochastic) are {}.\n'.format(max_policies_calc(value)))


def deterministic_solver():
    '''
        Solves the linear system of equations to find a solution for the grid.

        @params     none
        @returns    none
    '''
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  # current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r

    A_inv = np.linalg.inv(A)
    x = np.matmul(A_inv,b)
    #x = x - x[-1]
    #x = np.linalg.solve(A, b)
    draw_image(np.round(x.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.savefig('pa1_valueFunc_deterministic.png')
    # plt.show()
    plt.close()
    draw_policy(x.reshape(WORLD_SIZE, WORLD_SIZE))
    plt.savefig('pa1_optimal_policy_deterministic.png')
    # plt.show()
    plt.close()


def max_policies_calc(optimal_values):
    '''
        Calculates the max number of policies by looking for cells with equiprobable q values.
        For example if 2 cells have 2 different policies each, the total number of overall
        policies is 2*2 = 4

        @params     optimal values:np.matrix : 2D matrix containing q values for each cell

        @returns    max_policies:int         : The max number of policies
    '''
    max_policies = 1

    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        max_policies *= np.count_nonzero(next_vals == np.max(next_vals))

    return max_policies


if __name__ == '__main__':
    stochastic_solver()
    deterministic_solver()
    print('Please check the generated images saved as .png files for the policies. ' +
           'Apologies, but I thought it would be better to save them as pictures since ' +
            'the command line output would get skewed if window size was too small.')
