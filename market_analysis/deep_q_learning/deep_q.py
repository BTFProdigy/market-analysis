import math
import random

import numpy as np

INIT_EPSILON = 1
# EXP_DECAY_EPSILON = 5e-5
BATCH_SIZE = 50
MIN_EPSILON =0.01

class DeepQ:

    def __init__(self, neural_network, environment, replay_memory, statistics, num_of_actions, num_of_states):

        self.epsilon = 1
        self.num_of_iterations = 100
        self.gamma = 0.97
        self.num_of_actions, self.num_of_features = num_of_actions, num_of_states

        self.neural_network = neural_network
        self.environment = environment
        self.replay_memory = replay_memory
        self.statistics = statistics
        self.steps = 0

        self.exp_decay_epsilon = self.get_exp_decay_factor()

    def choose_action(self, state):
        if random.uniform(0,1) < self.epsilon:
            self.num_of_random_actions+=1
            return random.randint(0, self.num_of_actions - 1)

        else:
            return self.get_best_action_for_state(state)

    def get_best_action_for_state(self, state):
        return np.argmax(self.neural_network.predict(state))

    def iterate_over_states(self):
        for iteration in range(self.num_of_iterations):

            print iteration
            self.reset_for_iteration()

            reward, actions = self.one_iteration()
            self.statistics.add_epsilon(self.epsilon)
            self.statistics.add_reward(reward)
            self.statistics.actions = actions
            self.statistics.add_budget(self.environment.budget)
            self.statistics.add_stocks(self.environment.num_of_stocks)
            self.statistics.add_random_actions(self.num_of_random_actions)

            # self.print_q_values()
            if self.converged(iteration):
                break

    def converged(self, iteration):
        n = 10

        if iteration < n:
            return False

        last_n_rewards = self.statistics.rewards_history[-n:]
        previous_for_last_n_rewards = self.statistics.rewards_history[-n-1:-1]

        ratios = map(lambda x: x[0]/x[1],  zip(last_n_rewards, previous_for_last_n_rewards))
        threshold =  1e-3

        less_than_threshold = filter(lambda  x: x>threshold, ratios)
        return len(less_than_threshold) == 0

    def reset_for_iteration(self):
        self.environment.reset()
        self.num_of_random_actions = 0

    def one_iteration(self):
        total_reward = 0
        actions = []
        while True:
            state = self.environment.curr_state
            action = self.choose_action(state)
            actions.append(action)

            next_state, reward, done = self.environment.step(action)

            self.replay_memory.add((state, action, reward, next_state))
            self.replay()

            if done:
                break

            self.epsilon = self.update_epsilon()
            self.steps+=1

            total_reward += reward

        return total_reward, actions
    
    # za poslednju iteraciju ili ne
    def plot_state_action_reward(self):
        return

    def update_epsilon(self):
        return INIT_EPSILON * math.exp(-self.exp_decay_epsilon * self.steps)
        return MIN_EPSILON+(INIT_EPSILON-MIN_EPSILON) * math.exp(-self.exp_decay_epsilon * self.steps)

    def get_exp_decay_factor(self):
        step_size_per_iteration = self.environment.get_num_of_states_per_episode()
        total_num_of_steps = self.num_of_iterations*step_size_per_iteration

        return math.log(MIN_EPSILON)/-total_num_of_steps

    def print_q_values(self):
        batch = self.replay_memory.sample(BATCH_SIZE)

        states = np.array([experience_tuple[0] for experience_tuple in batch])
        q_predict_states = self.neural_network.predict_batch(states)
        print q_predict_states
        print q_predict_states[:, 0]

    def replay(self):
        batch = self.replay_memory.sample(BATCH_SIZE)
        states = np.array([exp_tuple[0] for exp_tuple in batch])

        next_states = np.array([(np.zeros(self.num_of_features)
                                if exp_tuple[3] is None
                                else exp_tuple[3]) for exp_tuple in batch])

        q_values = self.neural_network.predict_batch(states)
        q_values_next = self.neural_network.predict_batch(next_states)

        self.update_q_values_and_train_net(batch, q_values, q_values_next, states)

    def update_q_values_and_train_net(self, batch, q_values, q_values_next, states):
        for i, exp_tuple in enumerate(batch):
            state, action, reward, next_state = exp_tuple

            q_values[i][action] = (reward + self.gamma*np.max(q_values_next[i])
                                    if next_state is not None
                                    else reward)
        self.neural_network.train(states, q_values)

    # def test(self, test_environment):
    #
    #     total_reward = 0
    #     actions = []
    #
    #     while True:
    #         state = test_environment.curr_state
    #         action = self.get_best_action_for_state(state)
    #         actions.append(action)
    #         next_state, reward, done = test_environment.step(action)
    #         total_reward += reward
    #
    #         if done:
    #             break
    #         # total_reward += reward
    #
    #     self.statistics.add_budget(test_environment.budget)
    #     self.statistics.add_stocks(test_environment.num_of_stocks)
    #     return total_reward, actions




