import numpy as np

from market_analysis.deep_q_learning.action import Action


class RealTimeEnvironment:

    def __init__(self, reward, data_getter,  ticker, action_performer, agent_state, data_preprocessor):
        self.agent_state = agent_state
        self.ticker = ticker
        self.data_getter =data_getter
        self.reward = reward
        self.action_performer = action_performer
        self.preprocessor = data_preprocessor

        self.reset()

    def get_data(self):
        data = self.data_getter.get_data(self.ticker)
        # instance = data.ix[-1]
        self.last_timestamp = data.index
        return self.preprocessor.transform_price(data['Close'])

    def reset(self):
        data = self.get_data()

        # self.last_timestamp = data.index[-1]
        self.curr_state = self.create_state(data,
                                            self.agent_state.num_of_stocks,
                                            self.agent_state.budget)

    def create_state(self, data, num_of_stocks, budget):
        values = np.append(data, [num_of_stocks, budget])
        return values

    def step(self, action):
        instance = self.data_getter.get_data(self.ticker)

        if instance.name != self.last_timestamp:
            self.perform_action(action)
            new_state = self.get_new_state(self.curr_state, action, instance)
            self.curr_state = new_state

    def perform_action(self, action):
        self.action_performer.perform_action(self.ticker, action)
        return

    # def step(self, action):
    #     reward = self.reward.get_reward(self.curr_state, action)
        # new_state = None
        #
        # self.curr_state = new_state

        # return new_state, reward


    def get_new_state(self, state, action, instance):

        price = self.preprocessor.inverse_transform_price(state[0])

        if action == Action.Sell:
            # sell
            self.agent_state.profit += 0.85*price*1

            self.agent_state.num_of_stocks_sold+=1
            self.agent_state.budget+= 0.85*price*1
            self.agent_state.num_of_stocks-=1
        else:
            #     buy
            self.agent_state.num_of_stocks_bought+=1
            self.agent_state.profit -= 1.15*price*1

            self.agent_state.budget-= 1.15*price*1
            self.agent_state.num_of_stocks+=1

        return self.create_state(self.preprocessor.transform_price(instance['Close']),
                                 self.agent_state.num_of_stocks,
                                 self.agent_state.budget)
