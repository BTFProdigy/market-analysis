from market_analysis.deep_q_learning import config_getter
from market_analysis.deep_q_learning.data_api.db_worker import DBWorker
from market_analysis.deep_q_learning.environment.agent_state import AgentState
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

    def create_agents(self):
        num_stocks = 5
        budget = 10000

        agent_state1 = AgentState(num_stocks, budget)
        agent_state2 = AgentState(num_stocks, budget)
        agent_state3 = AgentState(num_stocks, budget)
        ticker = "BTC-EUR"

        agent1 = Trader(db_worker, "trained", False)
        agent2 = Trader(db_worker, "random1", True)
        agent3 = Trader(db_worker, "random2", True)


        agent1.trade(agent_state1, ticker, self.action_performer, "model_novo14/")
        # self.action_performer.perform_action("BTC-EUR", Action.Buy)
        agent2.trade(agent_state2, ticker, self.action_performer, "")
        agent3.trade(agent_state3, ticker, self.action_performer, "")

        return

def get_config(file):
    config = configparser.ConfigParser()
    # config.read('config')
    config.read(os.path.dirname(os.path.dirname(__file__)) + '/' + file)


config = config_getter.get_config('config')
db_address = config.getint('DB','db_address')
db_name = config.getint('DB','db_name')
db_worker = DBWorker(db_address, db_name)

local_agents = LocalAgents(db_worker)
local_agents.create_agents()







