import uuid

from market_analysis.deep_q_learning.action import Action
from market_analysis.deep_q_learning.data.order import Order
from market_analysis.deep_q_learning.local_agents_environment.action_performer import ActionPerformer

class OrderBookActionPerformer(ActionPerformer):
    def __init__(self, order_book):
        self.order_book = order_book

    def perform_action(self, ticker, action):
        if action != Action.DoNothing:
            side = "Ask" if action == Action.Sell else "Bid"
            order = Order(str(uuid.uuid4()), ticker, 0, 1, side)
            self.order_book.receive_order(order)
        return

    def refresh(self, order_book):
        self.order_book = order_book