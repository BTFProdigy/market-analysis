from market_analysis.deep_q_learning import config_getter, paths
from market_analysis.deep_q_learning.data_api.db_worker import DBWorker
from market_analysis.deep_q_learning.environment.agent_state import AgentState
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.data_getter.fake_real_time_trading_data_getter import \
    FakeRealTimeTradingDataGetter
from market_analysis.deep_q_learning.preprocessing.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reinforcement.reward import Reward
from market_analysis.deep_q_learning.trading.actions.order_book_action_performer import OrderBookActionPerformer
from market_analysis.deep_q_learning.trading.local_agents_environment.order_book import OrderBook

from market_analysis.deep_q_learning.trading.trader import Trader
import os.path
import configparser

class LocalAgents:

    def __init__(self, db_worker):
        self.order_book = OrderBook()
        self.db_worker = db_worker

        self.action_performer = OrderBookActionPerformer(self.order_book)
        self.refresh_orders("BTC-EUR")

    def refresh_orders(self, ticker):
        bids, asks = self.db_worker.get_latest_order_book(ticker)
        self.order_book.generate_bids_and_asks(bids, asks)

        self.action_performer.refresh(self.order_book)

    def create_agents(self, num_stocks, budget):
        agent_state1 = AgentState(num_stocks, budget)
        agent_state2 = AgentState(num_stocks, budget)
        agent_state3 = AgentState(num_stocks, budget)
        ticker = "BTC-EUR"

        agent1 = Trader(self.db_worker, "trained", False)
        agent2 = Trader(self.db_worker, "random1", True)
        agent3 = Trader(self.db_worker, "random2", True)

        agent1.trade(self.build_trading_env(agent_state1, ticker),"model_updated-12-12-12-february10-tanh/")
        agent2.trade(self.build_trading_env(agent_state2, ticker), "")
        agent3.trade(self.build_trading_env(agent_state3, ticker), "")

    def run(self):
        num_stocks = 5
        budget = 10000
        self.create_agents(num_stocks, budget)

    def build_trading_env(self, agent_state, ticker):
        data_preprocessor = DataPreprocessor.get_instance()
        data_preprocessor.load_scalers(paths.get_scalars_path())
        realtime_data_getter = FakeRealTimeTradingDataGetter(self.db_worker, data_preprocessor, '2018-11-24 23:00:00')

        reward = Reward()
        env_builder = EnvironmentBuilder(reward)

        env = env_builder.build_trading_environment(realtime_data_getter, ticker, self.action_performer, agent_state, data_preprocessor)
        return env









