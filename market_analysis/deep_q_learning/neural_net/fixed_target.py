class FixedTarget:

    def __init__(self, target, predicted):
        self.target = target
        self.predicted= predicted


    def copy_weights(self):
        weights = self.predicted.get_weights()
