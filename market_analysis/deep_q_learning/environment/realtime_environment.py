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

        return self.preprocessor.transform_price(data['Price'])

    def reset(self):
        self.last_timestamp = None
        data = self.get_data()

        # self.last_timestamp = data.index[-1]
        self.curr_state = self.create_state(data,
                                            self.preprocessor.transform_stocks(1),
                                            self.preprocessor.transform_budget(1),
                                            0)

    def create_state(self, data, num_of_stocks, budget, inventory):
        values = np.append(data, [num_of_stocks, budget, inventory])
        return values

    def step(self, action):
        instance = self.data_getter.get_data(self.ticker)

        if instance.name != self.last_timestamp:

            # if action == Action.Buy and self.agent_state.budget > instance or\
            #     action == Action.Sell and self.agent_state.num_of_stocks>0:
            self.perform_action(action)
            print 'Action perfomed: {}'.format(self.get_action(action))
            new_state = self.get_new_state(self.curr_state, action, instance)

            if action == Action.Sell:
                self.agent_state.remove_inventory()
            self.curr_state = new_state

            self.last_timestamp = instance.name

    def get_action(self, action):
        actions = ['Buy', 'Sell', 'Do Nothing']
        return actions[action]

    def perform_action(self, action):
        self.action_performer.perform_action(self.ticker, action)
        return

    def get_new_state(self, state, action, instance):

        price = self.preprocessor.inverse_transform_price(state[0])

        if action == Action.Sell:
            # sell

            self.agent_state.num_of_stocks_sold+=1
            self.agent_state.num_of_stocks-=1
            self.agent_state.profit += price*1
            self.agent_state.budget += price*1

            self.agent_state.profit_by_selling += price-self.agent_state.get_inventory() if \
                self.agent_state.get_inventory() != 0 else 0

        elif action == Action.Buy:
            #     buy
            self.agent_state.num_of_stocks_bought+=1
            self.agent_state.profit -= price*1
            self.agent_state.budget-= price*1
            self.agent_state.num_of_stocks+=1
            self.agent_state.inventory.append(price)

        return self.create_state(self.preprocessor.transform_price(instance['Price']),
                                 1 if self.agent_state.num_of_stocks>0 else 0,
                                 1 if self.agent_state.budget>0 else 0,
                                 # 0 if self.agent_state.get_inventory() == 0 else self.preprocessor.transform_price(self.agent_state.get_inventory())), False
                                 self.preprocessor.transform_price(self.agent_state.get_inventory()))
