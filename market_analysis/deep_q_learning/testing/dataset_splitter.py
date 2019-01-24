class DataSetSplitter:
    def split_train_test(self, data, train_set_size):
        size = data.shape[0]
        train_size = int(train_set_size*size)

        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]

        return train_data, test_data

    def split_train_validation_test(self, data, train_set_size, validation_set_size):
        size = data.shape[0]
        train_size = int(train_set_size*size)
        validation_size = int(validation_set_size*size)

        train_data = data.iloc[:train_size]
        validation_data = data.iloc[train_size:train_size+validation_size]
        test_data = data.iloc[train_size+validation_size:]

        return train_data, validation_data, test_data