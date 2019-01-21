from market_analysis.deep_q_learning.reinforcement.deep_q_improvement_decorator import DeepQImprovementDecorator


class FixedTarget(DeepQImprovementDecorator):

    def __init__(self, deep_q, target_net):
        super(FixedTarget, self).__init__(deep_q)
        self.target_net = target_net
        self.updating_target_freq = 400

    def replay(self):
        if self.epsilon_strategy.steps % self.updating_target_freq == 0:
            self.copy_weights()
        super(FixedTarget, self).replay()

    def copy_weights(self):
        weights, biases = self.neural_network.get_weights_and_biases()
        self.target_net.copy_weights_and_biases(weights, biases)

    def get_next_q_values(self,next_states):
        return self.target_net.predict_batch(next_states)
