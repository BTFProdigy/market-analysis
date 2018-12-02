from market_analysis.deep_q_learning.local_agents_environment.action_performer import ActionPerformer


class NullActionPerformer(ActionPerformer):
    def perform_action(self, ticker, action):
        pass