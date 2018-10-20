
class StateSpace:
    def __init__(self, num_of_features, number_of_parts_for_features):
        self.num_of_features = num_of_features
        self.bounds = []
        self.number_of_parts_for_features = number_of_parts_for_features

    def get_num_of_states(self):
        num_of_states = self.number_of_parts_for_features**self.num_of_features
        return num_of_states

    def get_states_encoded(self):
        num_of_states = self.get_num_of_states()
        return range(num_of_states)

    def set_bounds_for_features(self, mins, maxs, number_of_parts):
        zipped_min_max = zip(mins, maxs)

        for zipped in zipped_min_max:
            min = round(zipped[0], 4)
            # min = zipped[0]
            max = round(zipped[1], 4)
            # max = zipped[1]
            width = (max - min)/number_of_parts
            self.bounds.append(list(self.frange(min + width, max + width, width)))

    def frange(self, x, y, jump):
        while x < y:
            yield x
            x += jump

    def diskretize_features(self, features):
        maxs = features.max()
        maxs.round(4)
        mins = features.min()
        mins.round(4)

        self.set_bounds_for_features(mins, maxs, self.number_of_parts_for_features)

    def get_state(self, instance, owning_the_stock, budget):
        diskretized = []

        for index, feature in enumerate(instance.values):
            diskretized_value = self.diskretize(float(feature), index)
            diskretized.append(diskretized_value)

        return self.create_state_from_values(diskretized, owning_the_stock)

    def create_state_from_values(self, diskretized, owning_the_stock):
        basis = self.number_of_parts_for_features
        coded = [element*(basis**i) for (element, i) in zip(diskretized, range(self.num_of_features))]
        # return owning_the_stock*(basis**(number_of_features)) + sum
        return sum(coded)

    def diskretize(self, value, index):
        bounds = self.bounds[index]
        for i, bound in enumerate(bounds):
            if bound>value:
                diskretized = i
                return diskretized
        return i


