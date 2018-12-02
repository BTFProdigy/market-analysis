import cPickle
from threading import Timer
import numpy as np
from market_analysis.deep_q_learning.environment.environment_builder import EnvironmentBuilder
from market_analysis.deep_q_learning.environment.real_time_trading_data_getter import RealTimeTradingDataGetter
from market_analysis.deep_q_learning.data.db_worker import DBWorker
from market_analysis.deep_q_learning.neural_net.neural_network import NeuralNetwork
from market_analysis.deep_q_learning.environment.data_preprocessor import DataPreprocessor
from market_analysis.deep_q_learning.reward import Reward
import paths
class Trader:

    def __init__(self, name):
        self.name = name


    def trade(self, agent_state, ticker, action_performer, model):

        model_path = paths.get_models_path()+model
        db_worker = DBWorker()
        data_preprocessor = DataPreprocessor.get_instance()
        data_preprocessor.load_scalers(paths.get_scalars_path())
        realtime_data_getter = RealTimeTradingDataGetter(db_worker, data_preprocessor)

        reward = Reward(data_preprocessor)
        env_builder = EnvironmentBuilder(reward)

        env = env_builder.build_realtime_environment(realtime_data_getter, ticker, action_performer, agent_state, data_preprocessor)

        # path = "neural_net/model/"

        model_parameters = self.load_model_parameters(model_path + "parameters")
        nn = NeuralNetwork(model_parameters.input_size, model_parameters.output_size,
                           model_parameters.num_hidden_nodes1, model_parameters.num_hidden_nodes2,
                           model_parameters.activation_function1, model_parameters.activation_function2)
        nn = nn.restore_model(model_path + "model")

        t = Timer(0, self.perform_realtime_actions, args = (env, nn))
        t.start()
        return

    def perform_realtime_actions(self, env, nn):
        state = env.curr_state
        q_values = nn.predict(state)
        action = np.argmax(q_values)

        env.step(action)
        agent_state = env.agent_state
        print 'Agent {} has budget {} and number of stocks {}'.format(self.name, agent_state.budget, agent_state.num_of_stocks)
        return
        t = Timer(60, self.perform_realtime_actions(env, nn))
        t.start()

    def load_model_parameters(self, file_name):
        with open(file_name, 'rb') as file:
            return cPickle.load(file)