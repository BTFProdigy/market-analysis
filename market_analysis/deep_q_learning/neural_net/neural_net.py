import tensorflow as tf

import cPickle

from tensorflow.python.ops.losses.losses_impl import Reduction

from market_analysis.deep_q_learning.neural_net import ActivationFunction
from market_analysis.deep_q_learning.neural_net.model_parameters import ModelParameters
import matplotlib.pyplot as plt
import market_analysis.deep_q_learning.paths as paths

class NeuralNet:

    def __init__(self, num_of_states, num_of_actions, hidden_nodes, activation_functions):
        self.num_of_states = num_of_states

        self.hidden_nodes = hidden_nodes
        self.num_actions = num_of_actions

        self.activation_functions = activation_functions

        self.weights = [0]*4
        self.biases = [0]*4

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

        with tf.name_scope('layers'):

            acts = []
            num_hidden_nodes = len(hidden_nodes)
            for i in range(num_hidden_nodes):
                if i == 0:
                    input_size = num_of_states
                    input = self.states
                else:
                    input_size = hidden_nodes[i-1]
                    input = acts[i-1]

                self.weights[i] = tf.Variable(tf.random_uniform([input_size, hidden_nodes[i]], minval=-0.05, maxval=0.05), dtype=tf.float32)
                # self.biases[i] = tf.Variable(tf.random_uniform([hidden_nodes[i]]))
                act = self.get_activation_function(self.activation_functions[i])
                acts.append(act(tf.matmul(input, self.weights[i])))

                # self.weights[1] = tf.Variable(tf.random_uniform([hidden_nodes_layer1, hidden_nodes_layer2], minval=-0.05, maxval=0.05), dtype=tf.float32)
                # self.biases[1] = tf.Variable(tf.random_uniform([hidden_nodes_layer2]))
                #
                # self.weights[2] = tf.Variable(tf.random_uniform([hidden_nodes_layer2, 8], minval=-0.05, maxval=0.05), dtype=tf.float32)
                # self.biases[2] = tf.Variable(tf.random_uniform([8]))

            self.weights[num_hidden_nodes] = tf.Variable(tf.random_uniform([hidden_nodes[num_hidden_nodes-1], num_of_actions], minval=-0.05, maxval=0.05), dtype=tf.float32)
            self.biases[num_hidden_nodes] = tf.Variable(tf.random_uniform([num_of_actions]))

            # act1 = self.get_activation_function(self.activation_functions[0])
            # act2 = self.get_activation_function(self.activation_functions[1])
            # act3 = self.get_activation_function(self.activation_functions[2])

            # A1 = act1(tf.matmul(self.states, self.weights[0]))
            # A2 = act2(tf.matmul(A1, self.weights[1]) )
            # A3 = act3(tf.matmul(A2, self.weights[2]))
            self.predicted_q = tf.matmul(acts[num_hidden_nodes-1], self.weights[3])

            # tf.summary.histogram("predicted", self.predicted_q)

        with tf.name_scope("loss" +self.get_architecture_string()):
            self.loss = tf.losses.mean_squared_error(self.target_q, self.predicted_q)

            tf.summary.scalar("loss", self.loss)

        # global_step = tf.Variable(0, trainable=False)
        # lr= tf.train.exponential_decay(0.01, global_step,
        #                                            100000, 0.96, staircase=True)

        with tf.name_scope("training"):
            # self.optimizer = tf.train.ProximalAdagradOptimizer(
            #     learning_rate=0.01,
            #     l1_regularization_strength=0.01,
            #     l2_regularization_strength=0.01).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        # self.optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def get_weights_and_biases(self):
        return self.session.run([self.weights, self.biases])

    def copy_weights_and_biases(self, weights, biases):
        # w = weights[:]
        # # print self.weights
        # b = biases[:]
        for i in range(4):
            self.weights[i].assign(tf.Variable(weights[i]))
            self.biases[i].assign(tf.Variable(biases[i]))

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