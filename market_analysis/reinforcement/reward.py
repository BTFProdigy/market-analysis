class Reward:
    # update-uj
    def __init__(self, state_space, num_of_actions):
        states = state_space.get_states_encoded()
        self.fill_r_table(states, num_of_actions)

    def fill_r_table(self, states, num_of_actions):
        self.r_table = dict()

        for state in states:
            for action in range(num_of_actions):
                self.r_table[state] = dict()
                self.r_table[state][action] = self.get_reward(state, action)

    # ne valja
    def get_reward(self, state, action):

        if (state==0 or state == 1 or state==6 or state == 5) and action == 0:
            reward=200-100

        elif (state==0 or state == 1 or state==6 or state == 5) and action == 1:
            reward=-200+100

        elif (state==0 or state == 1 or state==6 or state == 5) and action == 2:
            reward=-200+100

        elif (state==18 or state==19 or state==23 or state==24) and action == 1:
            reward=200+100

        elif (state==18 or state==19 or state==23 or state==24) and action == 0:
            reward=-200-100

        elif (state==18 or state==19 or state==23 or state==24) and action == 2:
            reward=-200+100
        else:

            if action == 2:
                reward=1000

            if action == 1:
                reward=-200+100
            else:
                reward=-200-100
        return reward

    def get_sharpe_ratio_reward(self):
        return

    def get_short_term_rewards(self):
        return

    def get_long_term_rewards(self):
        return