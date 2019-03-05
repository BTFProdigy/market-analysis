from errno import EEXIST
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from market_analysis.deep_q_learning.reinforcement.action import Action

models_path = '/home/nissatech/Documents/Market Analysis Data/Plots/KaoNovo/'
class Evaluation:

    def plot_actions_during_time(self, price, actions, model= None):

        index = price.index[-len(actions):]
        actions = pd.DataFrame(data = actions, index = index, columns = ['Action'])
        data_and_actions = pd.concat([price, actions], axis = 1)
        data_and_actions.dropna(inplace=True)

        actions_buy = data_and_actions[actions.Action ==Action.Buy]
        actions_sell = data_and_actions[actions.Action==Action.Sell]

        ax = price.plot()
        ax.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)
        plt.plot(actions_buy['Price'], "ro", label = 'Buy')
        plt.plot(actions_sell['Price'], "go", label = 'Sell')
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.title('Actions for last iteration')
        plt.legend()

        self.save_figure(model, 'Actions During time')
        plt.show()

    def evaluate(self, deep_q_statistics, model = None):
        self.plot_budget(deep_q_statistics.budget_history, model)
        self.plot_stocks(deep_q_statistics.num_of_stocks_history, model)
        self.plot_epsilon(deep_q_statistics.epsilon_history, model)
        self.plot_rewards(deep_q_statistics.rewards_history, model)
        self.plot_rewards_avg(deep_q_statistics.avg_rewards_history, model)

        self.plot_random_actions(deep_q_statistics.random_actions, model)
        self.plot_budget_for_last_iteration(deep_q_statistics.budget_for_last_iteration, model)
        self.plot_stocks_for_last_iteration(deep_q_statistics.stocks_for_last_iteration, model)
        self.plot_all_actions(deep_q_statistics.all_actions, model)
        self.scatter_plot_rewards_for_instances_last_iteration(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration, model)
        # self.show_rewards_for_buy_and_sell_in_last_iteration(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration)
        self.scatter_plot_rewards_for_actions_last_iteration(deep_q_statistics.actions_for_last_iteration, deep_q_statistics.rewards_for_last_iteration, model)
        self.plot_states_3d(deep_q_statistics.states, deep_q_statistics.actions_for_last_iteration, model)

        self.price_and_inventory_with_actions(deep_q_statistics.states, deep_q_statistics.actions_for_last_iteration, model)

        self.plot_contour(deep_q_statistics.all_rewards, model)
        self.plot_rewards_for_last_iteration(deep_q_statistics.rewards_for_last_iteration, model)

    def plot_rewards(self, rewards, model):
        self.plot_statistics(rewards, "Reward", model)

    def plot_rewards_avg(self, rewards_avg, model):
        self.plot_statistics(rewards_avg, " Average Reward", model)

    def plot_stocks(self, stocks, model):
        self.plot_statistics(stocks, "Number of owned stocks", model)

    def plot_stocks_for_last_iteration(self, stocks, model):
        self.plot_statistics(stocks, "Stocks for last iteration", model)

    def plot_budget(self, budget, model):
        self.plot_statistics(budget, "Budget", model)

    def plot_budget_for_last_iteration(self, budget, model):
        self.plot_statistics(budget, "Budget for last iteration", model)

    def plot_epsilon(self, epsilon, model):
        self.plot_statistics(epsilon, "Epsilon", model)

    def plot_random_actions(self, random_actions, model):
        self.plot_statistics(random_actions, "Number of random actions", model)

    def plot_statistics(self, list, ylabel, model, xlabel = "Iteration"):
        x = range(list.__len__())
        plt.plot(x, list)

        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(ylabel)

        self.save_figure(model, ylabel)
        plt.show()

    def show_rewards_for_buy_and_sell_in_last_iteration(self, actions, rewards):

        for index, action in enumerate(actions):
            print 'Action {}, Reward {}'.format(action, rewards[index])
            plt.text(len%500, index, str(rewards[index]), fontsize=8)
        plt.show()

    def scatter_plot_rewards_for_instances_last_iteration(self, actions, rewards, model):
        COLORS = ['blue', 'm', 'orange']
        ACTIONS = ['Buy', 'Sell', 'Do Nothing']

        colors = map(lambda x: COLORS[int(x)], actions)
        instances = range(len(actions))
        plt.scatter(instances, rewards, c=colors, alpha=0.3,
                    cmap='viridis')
        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)

        legend_elements = [
            pylab.Line2D([0], [0], color=COLORS[i], lw=4, label=action) for i, action in enumerate(ACTIONS)]
        plt.legend(handles=legend_elements, loc='best')

        # plt.colorbar()  # show color scale
        plt.title('Rewards for specific actions in the last iteration')
        plt.xlabel('Instance')
        plt.ylabel('Reward')
        self.save_figure(model, 'Scatter plot 1')
        plt.show()

    def scatter_plot_rewards_for_actions_last_iteration(self, actions, rewards, model):

        plt.scatter(map(lambda x: int(x), actions), rewards, alpha=0.3,
                    cmap='viridis')
        plt.grid(color = 'gray', linestyle = '-', linewidth = 0.25, alpha = 0.5)

        plt.title('Rewards for specific actions in the last iteration')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        self.save_figure(model, 'Scatter plot 2')
        plt.show()

    def plot_contour(self,  all_rewards, model):
        n = np.array(all_rewards)
        print (n.shape)
        plt.matshow(n, interpolation = 'bilinear')
        # plt.title('Rewards by iterations')
        plt.xlabel('Instance')
        plt.ylabel('Iteration')
        plt.colorbar()
        self.save_figure(model, "All rewards")
        plt.show()


    def plot_rewards_for_last_iteration(self, rewards, model):
        self.plot_statistics(rewards, "Reward", "Instance", model)

    def print_final_state(self, statistics):
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(statistics.budget_history[-1],
                                     statistics.num_of_stocks_history[-1],
                                     statistics.rewards_history[-1])


    def plot_all_actions(self, all_actions, model):
        x = range(all_actions.__len__())
        buy = [actions[0] for actions in all_actions]
        sell = [actions[1] for actions in all_actions]
        do_nothing = [actions[2] for actions in all_actions]

        plt.plot(x, buy, label = 'Buy')
        plt.plot(x, sell, label = 'Sell')
        plt.plot(x, do_nothing, label = 'Do Nothing')

        plt.legend()
        self.save_figure(model, 'Actions by iterations')
        plt.show()

    def save_figure(self, model, plot_name):
        if model is not None:

            try:
                if not path.exists(models_path + model):
                    makedirs(models_path+model)
            except OSError as exc: # Python >2.5
                if exc.errno == EEXIST:
                    pass
                else: raise

            plt.savefig(models_path + model+plot_name+'.png')


    def plot_states_3d(self,  states, actions, model):
        states = np.array(states)
        COLORS = ['blue', 'm', 'orange']
        ACTIONS = ['Buy', 'Sell', 'Do Nothing']
        pca3d = PCA(n_components=3)

        reduced_3d = pca3d.fit_transform(states)
        print ('PCA Full variance ratio: {}'.format(pca3d.explained_variance_ratio_.cumsum()))

        fig = pylab.figure()
        ax = Axes3D(fig)

        colors = map(lambda x: COLORS[int(x)], actions)
        ax.scatter(reduced_3d[:, 0], reduced_3d[:, 1], reduced_3d[:, 2], c = colors)


        legend_elements = [
            pylab.Line2D([0], [0], color=COLORS[i], lw=4, label=action) for i, action in enumerate(ACTIONS)]
        plt.legend(handles=legend_elements, loc='best')

        plt.title('States in reduced 3d with actions')
        self.save_figure(model, 'States in reduced 3d state with actions')
        plt.show()


    def plot_states_2d(self,  states, actions, model):
        states = np.array(states)
        COLORS = ['blue', 'm', 'orange']
        ACTIONS = ['Buy', 'Sell', 'Do Nothing']
        pca2d = PCA(n_components=2)

        reduced_2d = pca2d.fit_transform(states)
        print ('PCA Full variance ratio: {}'.format(pca2d.explained_variance_ratio_.cumsum()))

        colors = map(lambda x: COLORS[int(x)], actions)
        plt.scatter(reduced_2d[:, 0], reduced_2d[:, 1], c = colors)


        legend_elements = [
            pylab.Line2D([0], [0], color=COLORS[i], lw=4, label=action) for i, action in enumerate(ACTIONS)]
        plt.legend(handles=legend_elements, loc='best')

        plt.title('States in reduced 2d with actions')
        self.save_figure(model, 'States in reduced 2d state with actions')
        plt.show()

    def price_and_inventory_with_actions(self, states, actions, model):
        states = np.array(states)
        COLORS = ['blue', 'm', 'orange']
        ACTIONS = ['Buy', 'Sell', 'Do Nothing']


        colors = map(lambda x: COLORS[int(x)], actions)
        plt.scatter(states[:, 0], states[:, 3], c = colors)

        legend_elements = [
            pylab.Line2D([0], [0], color=COLORS[i], lw=4, label=action) for i, action in enumerate(ACTIONS)]
        plt.legend(handles=legend_elements, loc='best')

        plt.xlabel('Price')
        plt.ylabel('Inventory')
        plt.title('Price and inventory with actions')
        self.save_figure(model, 'States in 3d with actions 1')
        plt.show()