class AgentState:
    def __init__(self, num_of_stocks, budget):
        self.initial_num_of_stocks = num_of_stocks
        self.initial_budget = budget
        self.reset()

    def reset(self):
        self.num_of_stocks = self.initial_num_of_stocks
        self.profit = 0
        self.budget = self.initial_budget

        self.num_of_stocks_bought=0
        self.num_of_stocks_sold = 0
        self.inventory = []

    def get_inventory(self):
        if self.inventory.__len__() == 0:
            return 0
        else:
            return self.inventory[-1]

    def remove_inventory(self):
        if self.inventory.__len__() == 0:
            return
        else:
            return self.inventory.pop()