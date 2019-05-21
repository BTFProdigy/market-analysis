import cPickle

import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction

from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.neural_net.model_parameters import ModelParameters


class NeuralNetworkKeras:

    def __init__(self, num_of_states, num_of_actions, hidden_nodes, activation_functions):
        self.num_of_states = num_of_states

        self.hidden_nodes = hidden_nodes
        self.num_actions = num_of_actions

        self.activation_functions = activation_functions

        self.num_actions = num_of_actions


        self.setup_net(num_of_states, hidden_nodes, num_of_actions)
        self.session =  tf.Session()
        self.saver = tf.train.Saver()
        self.session.run(tf.global_variables_initializer())

        self.losses = []

    def get_architecture_string(self):
        return '%i_%i_%s_%s' % (self.hidden_nodes[0], self.hidden_nodes[1], self.activation_functions[0], self.activation_functions[1])

    def setup_net(self, num_of_states, hidden_nodes, num_of_actions):
        self.states = tf.placeholder(shape=(None, num_of_states), dtype=tf.float32, name="states")
        self.target_q = tf.placeholder(dtype=tf.float32, shape=[None, num_of_actions], name="target_q" )
        layers = []
        with tf.name_scope('layers'):

            num_hidden_nodes = len(hidden_nodes)
            for i in range(num_hidden_nodes):
                if i == 0:
                    input = self.states

                else:
                    input = layers[i-1]

                layers.append(tf.layers.dense(input, hidden_nodes[i], activation=self.get_activation_function(self.activation_functions[i])))

            init = tf.random_uniform_initializer(minval=-0.05, maxval=0.05)
            self.predicted_q = tf.layers.dense(layers[num_hidden_nodes-1], num_of_actions, kernel_initializer=init)

            tf.summary.histogram("predicted", self.predicted_q)

        with tf.name_scope("loss" +self.get_architecture_string()):
            self.loss = tf.losses.mean_squared_error(self.target_q, self.predicted_q, reduction=Reduction.SUM)

            tf.summary.scalar("loss", self.loss)

        # global_step = tf.Variable(0, trainable=False)
        # lr= tf.train.exponential_decay(0.01, global_step,
        #                                            100000, 0.96, staircase=True)

        with tf.name_scope("training"):
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, states, target_q):
        _, loss = self.session.run([self.optimizer, self.loss], feed_dict={self.states: states, self.target_q: target_q})
        self.losses.append(loss)

    def get_activation_function(self, act_f):
        if act_f == ActivationFunction.Relu:
            return tf.nn.relu

        elif act_f == ActivationFunction.Tanh:
            return tf.nn.tanh

    def predict(self, state):
        return self.session.run(self.predicted_q, feed_dict={self.states:
                                                            state.reshape(1, self.num_of_states)})

    def predict_batch(self, states):
        predicted =  self.session.run(self.predicted_q, feed_dict={self.states: states})
        return predicted

    def save_model(self, path):
        self.saver.save(self.session, path)

        with open(path+"parameters", 'wb') as file:
            model = ModelParameters(self.hidden_nodes[0], self.hidden_nodes[1],
                                    self.activation_functions[0], self.activation_functions[1],
                                    self.num_of_states, self.num_actions)
            cPickle.dump(model, file)

    def restore_model(self, path):
        self.saver.restore(self.session, path)
        return self