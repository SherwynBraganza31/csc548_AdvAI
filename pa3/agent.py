import copy
import numpy as np
import math
import sys
import random
import gym
from matplotlib import pyplot as plt
from function_approximator import FourierBasis
from matplotlib import cm


class SarsaLambdaLinear:
    """
        Implements SARSA Lamda with Linear Function Approximation

        Attributes:
            actions - The size of the action space
            alpha - The learning rate
            gamma - The discount factor
            lamb - Lambda: the trace decay factor
            epsilon - Factor that decides whether to follow policy or explore
    """
    def __init__(self, function_approximator, actions: int,  alpha: float = 0.1,
                 gamma: float = 0.99, lamb: float = 0.9, epsilon: float = 0.05, initial_valfunc: float = 0.0):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.initial_valfunc = initial_valfunc
        self.function_approximator = None

        # if we don't have an approximation for each action, create deepcopies for each
        if type(function_approximator) != list or len(function_approximator) < self.actions:
            temp = []
            for x in range(self.actions):
                temp.append(copy.deepcopy(function_approximator))

            self.function_approximator = temp

        if initial_valfunc == 0.0:
            # initialize the 'theta' array corresponding to each action
            self.weights = np.zeros([self.function_approximator[0].getShape()[0], self.actions])
        else:
            self.weights = np.ones([self.function_approximator[0].getShape()[0], self.actions]) * initial_valfunc

        # initialize the weights for the trace update.
        self.lambda_weights = np.zeros(self.weights.shape)

    def traceClear(self):
        """
        Clear the weights of the trace vector
        """
        self.lambda_weights = np.zeros(self.weights.shape)

    def makeOnPolicy(self):
        """
        Makes the policy greedy
        """
        self.epsilon = 0

    def getStateActionVal(self, state, action):
        """
        Returns the Q value of the given state-action pair.
        """
        return np.dot(self.weights[:, action], self.function_approximator[action].getFourierBasisApprox(state))

    def getMaxStateActionVal(self, state):
        """
        Checks all Q values in
        """
        best = float('-inf')
        best_a = float('-inf')
        for a in range(0, self.actions):
            qval = self.getStateActionVal(state, a)
            if qval > best:
                best = qval
                best_a = a

        return best, best_a

    def next_move(self, state):
        """
        Stochastically returns an epsilon-greedy action from the current state.
        """

        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.actions)

        # Build a list of actions evaluating to max_a Q(s, a)
        best = float("-inf")
        best_actions = []

        for a in range(self.actions):
            thisq = self.getStateActionVal(state, a)

            if abs(thisq - best) < 0.001:
                best_actions.append(a)
            elif thisq > best:
                best = thisq
                best_actions = [a]

        if len(best_actions) == 0 or math.isinf(best):
            print("SarsaLambdaLinearFA: function approximator has diverged to infinity.", file=sys.stderr)
            return np.random.randint(0, self.actions)

        # Select randomly among best-valued actions
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action=None, terminal=False) -> float:
        """
            Runs a Sarsa update, given a transition.
            If no action is provided, it assumes an E-Greedy policy and finds
            an action that maximizes the Q value.

            Parameters
            ----------
            state - the state at time t
            action - the action to be taken to reach s+1
            reward - The reward received
            next_state - the state at t+1
            next_action - the action at t+1, if not present it is calculated
            terminal - if the next state is the terminal state.

            Returns
            -------
            delta - The TD error.

        """

        # Compute TD error
        delta = reward - self.getStateActionVal(state, action)

        # Only include s' if it is not a terminal state.
        if not terminal:
            if next_action is not None:
                delta += self.gamma*self.getStateActionVal(next_state, next_action)
            else:
                # adopt an exploration action
                (next_Q, next_action) = self.getMaxStateActionVal(next_state)
                delta += self.gamma * self.getMaxStateActionVal(next_Q)

        # Compute the basis functions for state s, action a.
        eval_f_action = self.function_approximator[action].getFourierBasisApprox(state)

        for each_a in range(0, self.actions):

            # Trace Update
            self.lambda_weights[:, each_a] *= self.gamma*self.lamb
            if each_a == action:
                self.lambda_weights[:, each_a] += eval_f_action

            # Weight Update
            self.weights[:, each_a] += self.alpha * \
                                       delta * \
                                       np.multiply(self.function_approximator[each_a].getGradientFactors(),
                                                   self.lambda_weights[:, each_a])

        # Return the TD error, which may be informative.
        return delta

def fourierBasis(env, samples: int = 10, episodes: int = 10, order: int = 3):
    gamma = 0.6
    run_data = np.zeros((samples, episodes, 2))

    state_dim = env.observation_space.shape[0]
    actions = env.action_space.n
    u_state = env.observation_space.high
    l_state = env.observation_space.low
    d_state = u_state - l_state
    best_learner = None
    best_sum = float('-inf')

    for sample in range(0, samples):

        fb = FourierBasis(order=order, dimensions=state_dim)
        learner = SarsaLambdaLinear(fb, actions=actions, gamma=gamma, lamb=1, epsilon=0.05, alpha=0.001)

        for episode in range(0, episodes):
            steps = 0
            # converge to pure on-policy for last 10 episodes
            if episode >= 0.8 * episodes:
                learner.makeOnPolicy()

            learner.traceClear()
            s = (env.reset() - l_state) / d_state
            a = learner.next_move(s)

            done = False
            nsteps = 0
            sum_r = 0.0

            while not done:
                sp, r, done, info = env.step(a)
                steps += 1
                sp = (sp - l_state) / d_state
                term = (done and not (info.get('TimeLimit.truncated', False)))
                ap = learner.next_move(sp)

                learner.update(s, a, r, sp, ap, terminal=term)
                s = sp
                a = ap
                sum_r += r * pow(gamma, nsteps)
                steps += 1

            run_data[sample, episode, :] = np.asarray([sum_r, steps])
            best_learner = learner if sum_r >= best_sum else best_learner

            # print('Run ' + str(sample + 1) + ", ep. " + str(episode + 1) + " return: " + str(sum_r) +
            #      ", # steps: " + str(steps))

        env.close()

    return run_data, best_learner

def runAnalysis():
    run_data = []
    learner = []
    basis = [3, 5, 7]
    colors = ['red', 'green', 'blue']
    episodes = 250
    samples = 50

    env = gym.make('MountainCar-v0')

    u_state = env.observation_space.high
    l_state = env.observation_space.low
    d_state = u_state - l_state

    for i in basis:
        temp_data, temp_learner = fourierBasis(env, samples, episodes, i)
        run_data.append(temp_data)
        learner.append(temp_learner)

    run_data_mean = []
    # run_data_std_dev = []
    data_range = range(0, episodes)

    for i in range(len(basis)):
        run_data_mean.append(np.mean(run_data[i], axis=0))
        # run_data_std_dev.append(np.std(run_data[i], axis=0))
        plt.plot(data_range[0::20], run_data_mean[i][0::20, 1], c=colors[i], label='order ' + str(basis[i]))

    # plotting steps taken to reach term vs. episodes
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.ylim(0, 395)
    plt.title('Steps Taken to Reach Terminal State vs. Episodes')
    plt.legend()
    plt.savefig('steps_vs_episodes.jpg')
    plt.close()

    # value function surface plots
    x_bounds, y_bounds = [l_state[0], u_state[0]], [l_state[1], u_state[1]]
    xs, ys = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], 100),
                         np.linspace(y_bounds[0], y_bounds[1], 100))
    zs = np.zeros(xs.shape)

    for base in range(len(basis)):
        for i in range(0, zs.shape[0]):
            for j in range(0, zs.shape[1]):
                s = [(xs[i, j] - l_state[0]) / d_state[0], (ys[i, j] - l_state[1]) / d_state[1]]
                (zq, _) = learner[base].getMaxStateActionVal(s)
                zs[i, j] = -1.0 * zq

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xs, ys, zs, cmap=cm.get_cmap("coolwarm"), linewidth=0, antialiased=False)
        ax.view_init(elev=45, azim=45)
        ax.set_xlabel('position')
        ax.set_ylabel('velocity')
        ax.view_init(elev=45, azim=45)
        ax.set_title('Cost to Go')
        fig.savefig('Cost to Go - Order ' + str(basis[base]) + '.jpeg')
        plt.close()

def runVisualDisplay():
    episodes = 250
    samples = 10
    env = gym.make('MountainCar-v0', render_mode='human').env
    results, learner = fourierBasis(env, samples=samples, episodes=episodes, order=5)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analysis':
            runAnalysis()
    else:
        runVisualDisplay()



