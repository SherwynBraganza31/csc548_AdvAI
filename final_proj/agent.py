import copy
import numpy as np
import gym
from matplotlib import pyplot as plt
from function_approximator import FourierBasis
import zmq
import sys

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


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
        best_action = 0

        for a in range(self.actions):
            thisq = self.getStateActionVal(state, a)

            if thisq >= best:
                best_action = a
                best = thisq

        # if len(best_actions) == 0 or math.isinf(best):
        #     print("SarsaLambdaLinearFA: function approximator has diverged to infinity.", file=sys.stderr)
        #     return np.random.randint(0, self.actions)

        # Select randomly among best-valued actions
        return best_action

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
        if next_action is not None:
            delta += self.gamma*self.getStateActionVal(next_state, next_action)
        else:
            # adopt an exploration action
            (next_Q, next_action) = self.getMaxStateActionVal(next_state)
            delta += self.gamma * self.getMaxStateActionVal(next_Q)
            print('No action found')

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

def fourierBasis(env, max_episodes: int = 300, samples: int = 10, order: int = 3, alpha=0.1, epsilon=0.0):
    gamma = 0.9
    run_data = np.zeros((samples, max_episodes))

    state_dim = env.observation_space.shape[0]  # grab state dimensions from sim
    actions = env.action_space.n  # grab possible actions from sim
    u_state = np.asarray([4.8, 5, 0.418, 5])  # construct tailored state bounds
    l_state = -1 * u_state
    d_state = u_state - l_state

    for sample in range(0, samples):
        fb = FourierBasis(order=order, dimensions=state_dim)
        learner = SarsaLambdaLinear(fb, actions=actions, gamma=gamma, lamb=0.9, epsilon=epsilon, alpha=alpha)

        ##############################################################
        # Terminate Episode when it reaches stability.
        # Stability = State is not terminated for more than 10000 steps
        ##############################################################
        stable = False
        episode = 0
        steps = 0
        while episode < max_episodes:
            steps = 0
            learner.traceClear()
            s = (env.reset() - l_state) / d_state
            # s = trim_state(s)
            a = learner.next_move(s)

            done = False
            stable = False
            sum_r = 0.0

            while not done:
                sp, r, done, info = env.step(a)
                # sp = trim_state(sp)

                if done:
                    r = -1
                else:
                    r = 0

                steps += 1
                sp = (sp - l_state) / d_state
                ap = learner.next_move(sp)

                learner.update(s, a, r, sp, ap)
                s = sp
                a = ap
                sum_r += r * pow(gamma, steps)
                steps += 1

                if steps >= 3000 and not done and not stable:
                    stable = True
                    done = True
                    # print(f'Run {sample}, Order {order} Reached Stability : {episode} episodes')

            run_data[sample, episode] = steps
            episode += 1

        env.close()

    return run_data

def trim_state(state):
    if state[1] > 3.0:
        state[1] = 3.0
    elif state[1] < -3.0:
        state[1] = -3.0
    if state[3] > 3.0:
        state[3] = 3.0
    elif state[3] < -3.0:
        state[3] = -3.0
    if state[0] > 2.4:
        state[0] = 2.4
    elif state[0] < -2.4:
        state[0] = -2.4
    if state[2] > 0.2095:
        state[2] = 0.2095
    elif state[2] < -0.2095:
        state[2] = -0.2095

    return state

def runAnalysis():
    run_data = []
    basis = [3, 5, 7, 9]
    colors = ['red', 'green', 'blue', 'black', 'magenta']
    episodes = 10
    samples = 1
    alpha = 0.001
    epsilon = 0.0

    env = gym.make('CartPole-v1').env

    for i in basis:
        temp_data = fourierBasis(env, episodes, samples, i, alpha=alpha, epsilon=epsilon)
        run_data.append(temp_data)

    run_data_mean = []
    # run_data_std_dev = []
    data_range = range(0, episodes)

    for i in range(len(basis)):
        run_data_mean.append(np.mean(run_data[i], axis=0))
        # run_data_std_dev.append(np.std(run_data[i], axis=0))
        plt.plot(data_range, run_data_mean[i], c=colors[i], label='order ' + str(basis[i]))

    # plotting steps taken to stable vs. episodes
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Steps Taken to Stable State vs. Episodes')
    plt.legend()
    plt.savefig(f'steps_vs_episodes_{alpha}.jpg')
    plt.close()

def runVisualDisplay():
    samples = 1
    env = gym.make('CartPole-v1', render_mode='human').env
    results = fourierBasis(env, max_episodes=5, samples=samples, order=7, alpha=0.001)
    plt.plot(np.arange(0, 5, 1), results, c='green', label='order 5')

    # plotting steps taken to stable vs. episodes
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Steps Taken to Stable State vs. Episodes')
    plt.legend()
    plt.savefig('steps_vs_episodes.jpg')
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analysis':
            runAnalysis()
    else:
        runVisualDisplay()




