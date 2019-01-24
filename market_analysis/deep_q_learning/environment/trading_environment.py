import numpy as np

from market_analysis.deep_q_learning.environment.environment import Environment
from market_analysis.deep_q_learning.reinforcement.action import Action


class TradingEnvironment(Environment):

    def __init__(self, reward, data_getter,  ticker, action_performer, agent_state, data_preprocessor):
        self.ticker = ticker
        self.data_getter = data_getter
        self.action_performer = action_performer

        super(TradingEnvironment, self).__init__(reward, data_preprocessor, agent_state)

    def get_data(self):
        data = self.data_getter.get_data(self.ticker)
        # instance = data.ix[-1]

        return self.preprocessor.transform_price(data['Price'])

    def reset(self):
        # self.last_timestamp = None
        data = self.get_data()

        self.curr_state = self.create_state(data,
                                            self.preprocessor.transform_stocks(1),
                                            self.preprocessor.transform_budget(1),
                                            0)

    def step(self, action):
        # instance = self.data_getter.get_data(self.ticker)

        is_new_data = self.data_getter.is_new_data_present(self.ticker)
        if is_new_data:
        # if instance.name != self.last_timestamp:

            # if action == Action.Buy and self.agent_state.budget > instance or\
            #     action == Action.Sell and self.agent_state.num_of_stocks>0:
            self.perform_action(action)
            print 'Action perfomed: {}'.format(self.get_action(action))
            new_state = self.get_new_state(self.curr_state, action)

            if action == Action.Sell:
                self.agent_state.remove_inventory()
            self.curr_state = new_state

            # self.last_timestamp = instance.name

    def get_action(self, action):
        actions = ['Buy', 'Sell', 'Do Nothing']
        return actions[action]

    def perform_action(self, action):
        self.action_performer.perform_action(self.ticker, action)
        return

    def get_new_state(self, state, action):
        price = self.preprocessor.inverse_transform_price(state[0])
        new_instance = self.data_getter.get_new_data(self.ticker)
        if action == Action.Sell:
            self.action_sell(price)
        elif action == Action.Buy:
            self.action_buy(price)

        return self.create_state(self.preprocessor.transform_price(new_instance['Price']),
                                 1 if self.agent_state.num_of_stocks>0 else 0,
                                 1 if self.agent_state.budget>0 else 0,
                                 # 0 if self.agent_state.get_inventory() == 0 else self.preprocessor.transform_price(self.agent_state.get_inventory())), False
                                 self.preprocessor.transform_price(self.agent_state.get_inventory()))