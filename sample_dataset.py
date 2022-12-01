import numpy as np

class create_samples():
    def __init__(self, X, y, n_samples = 1000) -> None:
        self.X = X 
        self.y = y 
        self.X_train = [] 
        self.y_train = []
        self.n_samples = n_samples

        self.create_sample()
        self.unison_shuffled_copies()
        
    def create_sample(self):
        np.random.seed(42)

        for i in np.unique(self.y):
            index = np.random.choice((self.y == i).nonzero()[0], size = 1000)
            y_ = self.y[index]

            X_ = self.X[index]
            self.y_train.append(y_)
            self.X_train.append(X_)
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        self.y_train = self.y_train.reshape(10*1000)
        self.X_train = self.X_train.reshape(10*1000, 784)

    def unison_shuffled_copies(self):
        np.random.seed(42)
        assert len(self.X_train) == len(self.y_train)
        p = np.random.permutation(len(self.X_train))

        return self.X_train[p], self.y_train[p]

