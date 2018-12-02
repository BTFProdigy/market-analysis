from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.environment.realtime_environment import RealTimeEnvironment
from train_environment import TrainEnvironment


class EnvironmentBuilder:

    def __init__(self, reward):
        self.reward = reward

    def build_train_environment(self, num_of_stocks, budget, data, data_preprocessor):

        agent_state = AgentState(num_of_stocks, budget)
        dataframe = data_preprocessor.preprocess_data(data, num_of_stocks, budget)

        environment = TrainEnvironment(self.reward, data['Close'].to_frame(), data_preprocessor, agent_state)
        return environment

    def build_realtime_environment(self, data_getter, ticker, action_performer, agent_state, data_preprocessor):
        return RealTimeEnvironment(self.reward, data_getter, ticker, action_performer, agent_state, data_preprocessor)


