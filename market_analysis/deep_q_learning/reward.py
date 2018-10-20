from market_analysis.deep_q_learning.action import Action
from collections import namedtuple

State = namedtuple('State', ['original_price', 'price', 'daily_return', 'bollinger_band_diff', 'stocks', 'budget'])

class Reward:

    #normal price, daily return, bollinger diff, price, num of stocks, budget

    def get_reward(self, state, action):
        state_tuple = State(*state)
        return self.reward_for_profit(state_tuple, action) + \
               self.get_reward_for_budget(state_tuple, action) +\
               self.get_penalty_for_selling_unexistent_stocks(state_tuple, action)

    def reward_if_state_good_enough(self, state):
        return

    def reward_for_profit(self, state, action):
        if action == Action.Sell:
            return state.bollinger_band_diff+state.daily_return+state.price
            # return state[0]
        elif action == Action.Buy:
            # return  -state[0]
            # pretvori u nagradu
            return -(state.bollinger_band_diff+state.daily_return)

        return state.bollinger_band_diff+state.daily_return+.7

    # ako zelimo na duze staze gledamo i volatility
    def reward_for_volatility(self, state, action):
        return

    def get_reward_for_budget(self, state, action):
        if state.budget <= state.original_price and action == Action.Buy:
            return -5
        elif state.budget > state.original_price and action == Action.Buy:
            return state.budget*0.001- state.stocks*0.01
        #     return 2
        return 0

    def get_penalty_for_selling_unexistent_stocks(self, state, action):
        if state.stocks <=0 and action == Action.Sell:
            return -5

        # reward according to number of stocks
        # initial budget utice na to
        elif state.stocks > 0 and action == Action.Sell:
            return state.stocks*0.05- state.budget*0.0001
        #     return 2
        return 0

    def get_sharpe_ratio_reward(self):
        return

    def sparse_reward(self):
        return