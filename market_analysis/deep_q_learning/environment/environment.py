import numpy as np

class Environment(object):

    def __init__(self, reward, preprocessor, agent_state):
        self.reward = reward

        self.agent_state = agent_state
        self.preprocessor = preprocessor
        self.num_of_features = None

        self.reset()

    def create_state(self, data, num_of_stocks, budget, inventory):
        values = np.append(data, [
            num_of_stocks,
            budget,
            inventory])
        self.num_of_features = len(values)
        return values

    def reset(self):
        pass

    def step(self, action):
        pass

    def get_new_state(self, state, action):
        pass

    def action_sell(self, price):
        self.agent_state.num_of_stocks_sold+=1
        self.agent_state.num_of_stocks-=1
        self.agent_state.profit += price*1
        self.agent_state.budget += price*1

        self.agent_state.profit_by_selling += price-self.agent_state.get_inventory() if \
            self.agent_state.get_inventory() != 0 else 0

    def action_buy(self, price):
        self.agent_state.num_of_stocks_bought+=1
        self.agent_state.profit -= price*1
        self.agent_state.budget-= price*1
        self.agent_state.num_of_stocks+=1
        self.agent_state.inventory.append(price)

    def get_agent_state(self):
        return self.agent_state

