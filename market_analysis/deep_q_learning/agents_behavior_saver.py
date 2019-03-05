import cPickle


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