from market_analysis.reinforcement.diskretizing_state_space import StateSpace

from market_analysis.data_reader import DataReaderImpl
from market_analysis.features import DateFrameBuilder
from market_analysis.reinforcement.evaluation import Evaluation
from market_analysis.reinforcement.q_table import QTable
from market_analysis.reinforcement.reward import Reward
from q_learning import QLearning


def build_dataframe(data):
    dataframe = (
        DateFrameBuilder(data)
                .add_returns()
                .add_bolinger_bands_diff(7)
            .build()
    )
    return dataframe

def start_learning(dataframe, data):
    q_learning = QLearning(0,0,0)
    evaluation = Evaluation()

    num_of_actions = 3
    state_space = StateSpace(2, 5)
    reward = Reward(state_space, 3)
    num_of_states = state_space.get_num_of_states()

    q_table = QTable(num_of_states, num_of_actions)
    q_learning.init(state_space, reward, q_table)
    q_learning.start_learning(3, dataframe)

    evaluation.plot_actions_during_time(data.Close, q_learning.actions)


def main():
    data_reader = DataReaderImpl()
    data = data_reader.read_data("/home/nissatech/Documents/Market Analysis Data/prices/Data/Stocks/",
                                 "aapl",
                                 '2016-06-24')
    if data.size != 0:
        dataframe = build_dataframe(data)
        dataframe.dropna(inplace=True)
        start_learning(dataframe, data)





