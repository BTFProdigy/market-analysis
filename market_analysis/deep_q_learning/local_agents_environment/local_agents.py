from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.data.db_worker import DBWorker
from market_analysis.deep_q_learning.environment.agent_state import AgentState
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

        num_stocks = 5
        budget = 10000

        agent_state1 = AgentState(num_stocks, budget)
        agent_state2 = AgentState(num_stocks, budget)
        agent_state3 = AgentState(num_stocks, budget)
        ticker = "BTC-EUR"

        agent1 = Trader("trained", False)
        agent2 = Trader("random1", True)
        agent3 = Trader("random2", True)


        agent1.trade(agent_state1, ticker, self.action_performer, "model_novo14/")
        # self.action_performer.perform_action("BTC-EUR", Action.Buy)
        agent2.trade(agent_state2, ticker, self.action_performer, "")
        agent3.trade(agent_state3, ticker, self.action_performer, "")

        return

local_agents = LocalAgents()
local_agents.create_agents()







