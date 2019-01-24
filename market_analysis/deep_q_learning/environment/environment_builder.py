from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.environment.batch_environment import BatchEnvironment
from market_analysis.deep_q_learning.environment.trading_environment import TradingEnvironment


class EnvironmentBuilder:

    def __init__(self, reward_func):
        self.reward_func = reward_func

    def build_batch_environment(self, num_of_stocks, budget, data, data_preprocessor):

        agent_state = AgentState(num_of_stocks, budget)
        dataframe = data_preprocessor.preprocess_data(data, num_of_stocks, budget)

        environment = BatchEnvironment(self.reward_func, data['Price'].to_frame(), data_preprocessor, agent_state)
        return environment

    def build_trading_environment(self, data_getter, ticker, action_performer, agent_state, data_preprocessor):
        return TradingEnvironment(self.reward_func, data_getter, ticker, action_performer, agent_state, data_preprocessor)


