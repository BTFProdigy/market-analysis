from market_analysis.deep_q_learning.deep_q_statistics import DeepQStatistics
from market_analysis.deep_q_learning.evaluation import Evaluation
import numpy as np

class Tester:

    def pass_throgh_data(self, env, nn):
        total_reward = 0
        actions = []
        while True:
            state = env.curr_state
            q_values = nn.predict(state)

            action = np.argmax(q_values)
            actions.append(action)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if done:
                break

        return total_reward, actions

    def test(self, nn, env):
        evaluation = Evaluation()
        total_reward, actions = self.pass_throgh_data(env, nn)

        print '''Budget: {},
                Num of stocks: {},
                Reward: {}'''.format(env.budget,
                                     env.num_of_stocks,
                                     total_reward)

        evaluation.plot_actions_during_time(env.original_close, actions)
        return total_reward, env.budget, env.num_of_stocks
