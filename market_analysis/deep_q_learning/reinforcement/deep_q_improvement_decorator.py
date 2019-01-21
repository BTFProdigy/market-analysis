from market_analysis.deep_q_learning.reinforcement.deep_q import DeepQ


class DeepQImprovementDecorator (DeepQ):
    def __init__(self, deep_q):
        # super.__init__(args)
        self.deep_q = deep_q

    def train(self):
        self.deep_q.train()



