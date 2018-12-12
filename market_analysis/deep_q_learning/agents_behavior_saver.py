import cPickle
import numpy as np
import pandas as pd
class AgentsBehaviorSaver:
    def save(self, actions, rewards):
        with open("behavior", "wb") as fp:   #Pickling
            cPickle.dump(actions, fp)
            cPickle.dump(rewards, fp)

    def load(self, path):
        with open(path + "behavior", "rb") as fp:   # Unpickling
            actions = cPickle.load(fp)
            rewards = cPickle.load(fp)

            return actions, rewards
            # return pd.Series(range(0, 50), name = 'Actions'), rewards
