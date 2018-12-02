from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.data.db_worker import DBWorker
from market_analysis.deep_q_learning.local_agents_environment.order_book import OrderBook
from market_analysis.deep_q_learning.local_agents_environment.order_book_action_performer import \
    OrderBookActionPerformer
from market_analysis.deep_q_learning.trader import Trader


class LocalAgents:

    def __init__(self):
        self.order_book = OrderBook()
        self.db_worker = DBWorker()

        self.action_performer = OrderBookActionPerformer(self.order_book)
        self.refresh_orders("BTC-EUR")

    def refresh_orders(self, ticker):
        bids, asks = self.db_worker.get_latest_order_book(ticker)
        self.order_book.generate_bids_and_asks(bids, asks)

        self.action_performer.refresh(self.order_book)

    def create_agents(self):

        num_stocks = 20
        budget = 1000
        ticker = "BTC-EUR"

        agent1 = Trader("trained")
        agent2 = Trader("random1")
        agent3 = Trader("random2")


        # agent1.trade(budget, num_stocks, ticker, self.action_performer, "")
        self.action_performer.perform_action("BTC-EUR", Action.Buy)
        # agent2.trade(budget, num_stocks, ticker, self.action_performer, "neural_net/model2/")
        # agent3.trade(budget, num_stocks, ticker, self.action_performer, "neural_net/model3/")

        return

local_agents = LocalAgents()
local_agents.create_agents()







