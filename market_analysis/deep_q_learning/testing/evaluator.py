import numpy as np
import pandas as pd
from market_analysis.deep_q_learning.agents_behavior_saver import AgentsBehaviorSaver
from market_analysis.deep_q_learning.evaluation.evaluation import Evaluation


class Evaluator:

    def pass_through_data(self, env, nn):

        total_reward = 0
        actions = []
        rewards = []

        while True:
            state = env.curr_state

            q_values = nn.predict(state)

            action = np.argmax(q_values)


            actions.append(action)

            next_state, reward, done = env.step(action)
            rewards.append(reward)
            total_reward += reward

            if done:
                break


        num = len(actions)
        all_actions=(float(actions.count(0))/num, float(actions.count(1))/num, float(actions.count(2))/num)

        return total_reward, total_reward/(float) (len(rewards)), actions, rewards, all_actions

    def test(self, model, env, data):
        evaluation = Evaluation()

        total_reward, avg_reward, actions, rewards, all_actions = self.pass_through_data(env, model)

        # self.save_agents_behavior(actions, rewards, data.index)
        agent_state = env.agent_state
        print '''Budget: {},
                Num of stocks: {},
                Reward: {}, 
                Avg reward:{},
                actions:{}, 
                profit:{}'''.format(agent_state.budget,
                                     agent_state.num_of_stocks,
                                     total_reward,
                                     avg_reward,
                                     all_actions, env.agent_state.profit_by_selling)

        evaluation.plot_actions_during_time(data['Price'], actions)
        return total_reward, agent_state.budget, agent_state.num_of_stocks

    # def save_agents_behavior(self, actions, rewards, index):
    #     behavior_saver = AgentsBehaviorSaver()
    #     actions_series = pd.Series(actions, index = index, name='Actions')
    #     rewards_series = pd.Series(rewards, index = index, name = 'Rewards')
    #     behavior_saver.save(actions_series, rewards_series)
