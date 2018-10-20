import tensorflow as tf


class NeuralNetwork:

    def __init__(self, num_of_states, num_of_actions, hidden_nodes_layer1, hidden_nodes_layer2, learning_rate):
        self.num_of_states = num_of_states

        self.setup_net(num_of_states, hidden_nodes_layer1, hidden_nodes_layer2, num_of_actions, learning_rate)

        self.session =  tf.Session()
        self.session.run(tf.global_variables_initializer())

    def setup_net(self, num_of_states, hidden_nodes_layer1, hidden_nodes_layer2, num_of_actions, learning_rate):
        self.states = tf.placeholder(shape=(None, num_of_states), dtype=tf.float32, name="states")
        self.target_q = tf.placeholder(dtype=tf.float32, shape=[None, num_of_actions], name="target_q" )

        layer1 = tf.layers.dense(self.states, hidden_nodes_layer1, activation=tf.nn.sigmoid)
        layer2 = tf.layers.dense(layer1, hidden_nodes_layer2, activation=tf.nn.sigmoid)
        self.predicted_q = tf.layers.dense(layer2, num_of_actions)

        self.loss = tf.losses.mean_squared_error(self.target_q, self.predicted_q)

        global_step = tf.Variable(0, trainable=False)
        lr= tf.train.exponential_decay(learning_rate, global_step,
                                                   100000, 0.96, staircase=True)

        self.optimizer = tf.train.GradientDescentOptimizer(lr).minimize(self.loss, global_step=global_step)

    def train(self, states, target_q):
        self.session.run(self.optimizer, feed_dict={self.states: states, self.target_q: target_q})


    def predict(self, state):
        return self.session.run(self.predicted_q, feed_dict={self.states:
                                                            state.reshape(1, self.num_of_states)})

    def predict_batch(self, states):

        return self.session.run(self.predicted_q, feed_dict={self.states: states})

