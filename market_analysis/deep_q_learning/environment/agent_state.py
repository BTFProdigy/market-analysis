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

        self.profit_by_selling = 0

    def get_inventory(self):
        if self.inventory.__len__() == 0:
            return 0
        else:
            return self.inventory[-1]

    def remove_inventory(self):
        if self.inventory.__len__() == 0:
            return
        else:
            self.inventory.pop(-1)

    # def add_action(self, is_buy_or_sell):
    #     self.period_tracker+=1
    #     if self.period_tracker == self.period:
    #         self.period_tracker = 0
    #
    #     if is_buy_or_sell:
    #         self.num_of_actions+=1

    def get_inv_len(self):
        return  self.inventory.__len__()