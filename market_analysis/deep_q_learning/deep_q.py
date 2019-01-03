import math
import random

import numpy as np
import matplotlib.pyplot as plt
from market_analysis.deep_q_learning.neural_net.neural_net_logger import NeuralNetLogger

BATCH_SIZE = 50

class DeepQ:

    def __init__(self, neural_network, environment, replay_memory, statistics, num_of_actions, num_of_states, epsilon_strategy, num_of_iterations, target_net):

        self.num_of_iterations = num_of_iterations
        self.gamma = 0.95
        self.num_of_actions, self.num_of_features = num_of_actions, num_of_states
        self.iteration = 0

        self.neural_network = neural_network
        self.environment = environment
        self.replay_memory = replay_memory
        self.statistics = statistics
        self.epsilon_strategy = epsilon_strategy

        self.neural_net_logger = NeuralNetLogger(neural_network)
        # self.epsilon = 1
        self.target_neural_network = target_net
        self.updating_target_freq = 400

    def choose_action(self, state):
        self.epsilon = self.epsilon_strategy.get_epsilon_2()
        if random.uniform(0,1) < self.epsilon:
            self.num_of_random_actions+=1
            return random.randint(0, self.num_of_actions - 1)
        else:
            return self.get_best_action_for_state(state)

    def get_best_action_for_state(self, state):
        return np.argmax(self.neural_network.predict(state))

    def print_actions(self, actions):
        print "Num buy: {}, Num sell: {}".format(actions.count(0), actions.count(1))

    def iterate_over_states(self):
        for iteration in range(self.num_of_iterations):
            self.iteration = iteration
            print 'Iteration {}'.format(iteration)

            self.reset_for_iteration()

            rewards, actions, budget, stocks = self.one_iteration()
            self.print_actions(actions)
            total_reward = sum(rewards)
            self.statistics.rewards_for_last_iteration = rewards
            self.statistics.budget_for_last_iteration = budget

            self.statistics.stocks_for_last_iteration = stocks

            self.statistics.add_epsilon(self.epsilon)
            self.statistics.add_reward(total_reward)
            self.statistics.add_reward_avg(float(total_reward)/len(rewards))
            self.statistics.actions_for_last_iteration = actions

            budget_at_the_end_of_iteration = budget[-1]
            stocks_at_the_end_of_iteration = stocks[-1]

            print 'Budget: {}, stocks: {}, Reward: {}, Random actions: {}, Profit: {}'.format(budget_at_the_end_of_iteration,
                                                                                  stocks_at_the_end_of_iteration,
                                                                                  total_reward,
                                                                                  self.num_of_random_actions,
                                                                                              self.environment.agent_state.profit_by_selling)
            self.statistics.add_budget(budget_at_the_end_of_iteration)
            self.statistics.add_stocks(stocks_at_the_end_of_iteration)
            self.statistics.add_random_actions(self.num_of_random_actions)

            # print self.neural_network.losses[-1]
            # self.print_q_values()
            # if self.converged(iteration):
            #     break

        plt.plot(self.neural_network.losses)
        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)
        plt.title('Loss function of neural network')
        plt.ylabel('Loss')
        plt.xlabel('Training step')
        plt.show()

    def converged(self, iteration):
        n = 20

        if iteration < n:
            return False

        last_n_rewards = self.statistics.rewards_history[-n:]
        previous_for_last_n_rewards = self.statistics.rewards_history[-n-1:-1]

        ratios = map(lambda x: x[0]/x[1] if x[1] != 0 else 0, zip(last_n_rewards, previous_for_last_n_rewards))
        threshold =  1e-4

        less_than_threshold = filter(lambda  x: x>threshold, ratios)
        return len(less_than_threshold) == 0

    def reset_for_iteration(self):
        self.environment.reset()
        self.num_of_random_actions = 0

    def one_iteration(self):

        rewards = []
        actions = []
        budget = []
        stocks = []

        while True:
            state = self.environment.curr_state
            action = self.choose_action(state)
            # print '{}, Iteration {}'.format(action, self.iteration)
            actions.append(action)

            next_state, reward, done = self.environment.step(action)

            self.replay_memory.add((state, action, reward, next_state))
            self.replay()
            rewards.append(reward)

            agent_state = self.environment.agent_state
            budget.append(agent_state.budget)
            stocks.append(agent_state.num_of_stocks)

            if done:
                break

        return rewards, actions, budget, stocks

    def print_q_values(self):
        batch = self.replay_memory.sample(BATCH_SIZE)

        states = np.array([experience_tuple[0] for experience_tuple in batch])
        q_predict_states = self.neural_network.predict_batch(states)
        print q_predict_states
        print q_predict_states[:, 0]

    def replay(self):
        # if self.epsilon_strategy.steps % self.updating_target_freq == 0:
        #     self.copy_weights()
        if BATCH_SIZE<= self.replay_memory.get_size():
            batch = self.replay_memory.sample(BATCH_SIZE)
            states = np.array([exp_tuple[0] for exp_tuple in batch])

            # organized_batch = list(zip(*batch))
            # states = np.array(organized_batch[0])
            next_states = np.array([(np.zeros(self.num_of_features)
                                    if exp_tuple[3] is None
                                    else exp_tuple[3]) for exp_tuple in batch])

            q_values = self.neural_network.predict_batch(states)

            # if self.iteration % 20== 0:
            #     print q_values

            q_values_next = self.neural_network.predict_batch(next_states)

            if self.iteration %10 == 0:
                print q_values

            self.update_q_values_and_train_net(states, batch, q_values, q_values_next)



            # print np.array(self.neural_network.weights[0].eval(session = self.neural_network.session))
            # print self.target_neural_network.weights[0]
            # print self.neural_network.weights[0].eval(session = self.neural_network.session) == self.target_neural_network.weights[0]

    def update_q_values_and_train_net(self, states, batch, q_values, q_values_next):
        grouped_batch = zip(*batch)

        states = np.array(grouped_batch[0])
        actions = np.array(grouped_batch[1])
        rewards = np.array(grouped_batch[2])

        p = np.amax(q_values_next, axis = 1)
        full_rewards = rewards+self.gamma*np.amax(q_values_next, axis = 1)

        q_values[range(len(states)), list(actions)] = full_rewards

        self.neural_network.train(states, q_values)

        # for i, exp_tuple in enumerate(batch):
        #     state, action, reward, next_state = exp_tuple
        #
        #     q_values[i][action] = (reward + self.gamma*np.max(q_values_next[i])
        #     if next_state is not None
        #     else reward)
        #     self.neural_network.train(states, q_values)
        # self.neural_net_logger.log_performance(states, q_values, self.iteration)

    def copy_weights(self):
        weights, biases = self.neural_network.get_weights_and_biases()
        self.target_neural_network.copy_weights_and_biases(weights, biases)

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




