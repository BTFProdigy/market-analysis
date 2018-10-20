
import numpy as np

class QTable:

    def __init__(self, num_of_states, num_of_actions):
        self.num_of_states = num_of_states
        self.num_of_actions = num_of_actions
        self.q_table = np.zeros((num_of_states, num_of_actions))

    def update_q(self, state, action, new_value):
        self.q_table[state, action] = new_value

    def get_value(self, state, action):
        return self.q_table[state, action]

    def get_max_value_for_state(self, state):
        return np.max(self.q_table[state])

    def get_best_action_for_state(self, state):
        return np.argmax(self.q_table[state])

    def get_random_action(self):
        random_action = np.random.randint(0, self.num_of_actions)
        return random_action