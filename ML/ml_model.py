
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# from sklearn.metrics import accuracy_score
from sortedcontainers import SortedList
import math
from scipy.stats import multivariate_normal as mvn


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v', '>', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'green', 'pink')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        # X_test, y_test = X[test_idx, :], y[test_idx]
        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')


class ml_model(object):
    """
    Parameters
    ------------
    method : str
        which ML lib
    learning_rate : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    num_epochs : int (default: 100)
        How many run to train dataset.
    shuffle : bool (default: True)
        Shuffles training data every epoch

    Attributes
    ------------
    w_ : 1d-array, float
        Weights
    errors_ : list, float
    """

    def __init__(self, method=None, learning_rate=0.01,
                 num_epochs=100, shuffle=None):
        self.method = method
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.w_ = []
        self.cost_ = []
        # print("ml_model: __init__")

    def fit(self, X_train, Y, standardize=False):

        # Dimensions, Features of X_Train
        self.N, self.D = X_train.shape
        print("X_train has {0} samples with {1} \
            features".format(self.N, self.D))

        self.Y = Y

        # print("X_train:", X_train)

        if(standardize is True):
            X_std = np.copy(X_train)
            for i in range(self.D):
                X_std[:, i] = \
                    (X_train[:, i] - X_train[:, i].mean()) / X_train[:, i].std()

            self.X_train = np.copy(X_std)

        else:
            self.X_train = np.copy(X_train)

        # print("X_train_std:", self.X_train)

        if (self.shuffle is True):
            # init w_ as random numbers
            self.w_ = np.random.randn(1 + self.D)

            # shuffle X_train, Y
            # r = np.random.permutation(self.N)
            # self.X_train = self.X_train[r]
            # self.Y = self.Y[r]

        else:
            # init w_ as zeros with w0
            self.w_ = np.zeros((1 + self.D))

        print("shape of X_train:", self.X_train.shape)
        print("shape of w_:", self.w_.shape)
        print("shape of Y:", self.Y.shape)

    def activation_fn(self, z, activation=None):

            if(activation is None):
                return z
            elif(activation == "sign"):
                return np.piecewise(z, [z < 0, z >= 0], [-1, 1])
            elif(activation == "step"):
                return np.piecewise(z, [z < 0, z >= 0], [0, 1])
            elif(activation == "sigmoid"):
                return 1.0 / (1.0 + np.exp(-z))

    def net_input(self, X_data):
        # net input: w0x0 + w1x1... + wixi
        # print("net_input:")
        # print("shape of X_data:", X_data.shape)
        # print("shape of self.w_[1:]:", self.w_[1:].shape)
        return np.dot(X_data, self.w_[1:]) + self.w_[0]

    def predict(self):
        pass

    def r2_evl(self, Y, Y_hat):
        d1 = Y - Y_hat
        d2 = Y - Y.mean()
        # print("Shape of d1:", d1.shape)
        # print("Shape of d2:", d2.shape)
        r2 = 1 - (d1.dot(d1) / d2.dot(d2))
        print("R2:", r2)

    def score(self, Y, Y_hat):
        print('Misclassified samples: %d' % (Y != Y_hat).sum())
        # print('Accuracy: %.2f' % accuracy_score(Y, Y_hat))
        print('Score:', np.mean(Y_hat == Y))


class myPerceptron(ml_model):

    def __init__(self, method=None, learning_rate=0.01,
                 num_epochs=100, shuffle=True, activation="sign"):
        # print("myPerceptron: __init__")
        super().__init__(method, learning_rate, num_epochs, shuffle)
        self.activation = activation

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y, standardize=False):
        super().fit(X_train, Y, standardize)
        self.errors_ = []
        for epoch in range(self.num_epochs):
            errors = 0
            for xi, y in zip(self.X_train, Y):
                y_hat = self.predict(xi)
                error = y - y_hat
                update = self.learning_rate * error
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(error != 0.0)
            self.errors_.append(errors)

        print("final w:\n", self.w_, "\nepochs:",
              (epoch+1), "/", self.num_epochs)
        plt.plot(range(1, len(self.errors_) + 1),
                 self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of erros')
        plt.show()

    def predict(self, X_test, standardize=False):
        print("shape of X_test:", X_test.shape)
        if(standardize is True):
            for i in range(self.D):
                X_test[:, i] = \
                 (X_test[:, i] - X_test[:, i].mean()) / X_test[:, i].std()

        z = self.net_input(X_test)
        # print("z:",z)
        Y_hat = self.activation_fn(z, self.activation)
        # print("Y_hat:",Y_hat)
        return Y_hat


class myAdaline(ml_model):

    def __init__(self, method=None, learning_rate=0.0001,
                 num_epochs=200, shuffle=True, mini_batch=False):
        # print("myAdaline: __init__")
        super().__init__(method, learning_rate, num_epochs, shuffle)
        self.mini_batch = mini_batch

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation_fn(self.net_input(xi), activation=None)
        error = (target - output)
        # print("shape of xi:", xi.shape)
        # print("shape of error:", error.shape)
        self.w_[1:] += self.learning_rate * xi.dot(error)
        self.w_[0] += self.learning_rate * error
        cost = 0.5 * error**2
        return cost

    def fit(self, X_train, Y, standardize=False):
        super().fit(X_train, Y, standardize)
        cost = 0
        avg_cost = 0
        for epoch in range(self.num_epochs):
            if(self.mini_batch is False):
                z = self.net_input(self.X_train)
                output = self.activation_fn(z, activation=None)
                error = (self.Y - output)
                self.w_[1:] += self.learning_rate * \
                    np.dot(self.X_train.T, error)
                self.w_[0] += self.learning_rate * error.sum()
                cost = (error**2).sum() / 2.0
                self.cost_.append(cost)
            else:
                cost = []
                for xi, target in zip(X_train, Y):
                    cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(Y)
                self.cost_.append(avg_cost)

        if(self.mini_batch is False):
            print("final w:\n", self.w_, "\nFinal cost:\n", cost,
                  "\nepochs:\n", (epoch+1), "/", self.num_epochs)
        else:
            print("final w:\n", self.w_, "\nFinal cost:\n", avg_cost,
                  "\nepochs:\n", (epoch+1), "/", self.num_epochs)

        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Sum-squared-error')
        plt.title('Adaline - Learning rate: {0}'.format(self.learning_rate))
        plt.tight_layout()
        plt.show()

    def partial_fit(self, X_train, Y):
        """Fit training data without reinitializing the weights"""
        if Y.ravel().shape[0] > 1:
            for xi, target in zip(X_train, Y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X_train, Y)
        return self

    def predict(self, X_test, standardize=False):
        print("shape of X_test:", X_test.shape)
        if(standardize is True):
            for i in range(self.D):
                X_test[:, i] = (X_test[:, i] - X_test[:, i].mean()) / X_test[:, i].std()

        z = self.net_input(X_test)
        # output = self.activation_fn(z, activation=None)
        Y_hat = self.activation_fn(z, activation="sign")
        return Y_hat


class myLogistic(ml_model):

    def __init__(self, method=None, learning_rate=0.0001,
                 num_epochs=200, shuffle=True):
        # print("myLogistic: __init__")
        super().__init__(method, learning_rate, num_epochs, shuffle)

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y, standardize=False):
        super().fit(X_train, Y, standardize)
        cost = 0
        for epoch in range(self.num_epochs):
            z = self.net_input(self.X_train)
            # print("z:", z)
            output = self.activation_fn(z, activation='sigmoid')
            # print("Shape of output:", output.shape)
            # print("output:", output)
            error = (self.Y - output)
            self.w_[1:] += self.learning_rate * np.dot(self.X_train.T, error)
            self.w_[0] += self.learning_rate * error.sum()
            cost = -Y.dot(np.log(output)) - ((1 - Y).dot(np.log(1 - output)))
            self.cost_.append(cost)

        print("final w:\n", self.w_, "\nFinal cost:\n", cost,
              "\nepochs:\n", (epoch+1), "/", self.num_epochs)

        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Sum-squared-error')
        plt.title('Adaline - Learning rate: {0}'.format(self.learning_rate))
        plt.tight_layout()
        plt.show()

    def predict(self, X_test, standardize=False):
        print("shape of X_test:", X_test.shape)
        if(standardize is True):
            for i in range(self.D):
                X_test[:, i] = (X_test[:, i] - X_test[:, i].mean()) / X_test[:, i].std()
        z = self.net_input(X_test)
        # output = self.activation_fn(z, activation='sigmoid')
        Y_hat = self.activation_fn(z, activation="step")
        return Y_hat


class myKNN(ml_model):

    def __init__(self, K):
        self.K = K

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y):
        self.X_train = X_train
        self.Y = Y
        # print("self.Y:", self.Y)
        # print("self.X_train:", self.X_train)

    def predict(self, X_test):
        print("shape of X_test:", X_test.shape)
        Y_pred = np.zeros((X_test.shape[0]))
        # print("X_test:", X_test)
        for i, xi_test in enumerate(X_test):
            sl = SortedList()
            for j, xi_train in enumerate(self.X_train):
                diff = xi_test - xi_train
                dist = math.sqrt(diff.dot(diff))
                if(len(sl) < self.K):
                    sl.add((dist, self.Y[j]))
                else:
                    if(dist < sl[-1][0]):
                        del sl[-1]
                        sl.add((dist, self.Y[j]))
            # print("sl:", sl)

            # vote
            votes = {}

            for _, v in sl:
                # print("v:", v)
                votes[v] = votes.get(v, 0) + 1
            max_votes = 0
            max_votes_class = -1
            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            Y_pred[i] = max_votes_class
        return Y_pred


class myBayes(ml_model):

    def __init__(self, naive=True, pdf='gaussian'):
        self.naive = naive
        self.pdf = pdf

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y, smoothing=1e-2):
        self.X_train = X_train
        self.Y = Y

        # print("self.Y:", self.Y)
        # print("self.X_train:", self.X_train)
        N, D = X_train.shape
        print("X_train has {0} samples with {1} features".format(
            N, D))

        self.lable_Prob_in_Y = {}
        labels = set(Y)
        self.lable_numbers = len(labels)

        if self.pdf is 'gaussian':
            if self.naive is True:
                self.X_train_mean_var_in_Y = {}
                for c in labels:
                    such_X_train = X_train[Y == c]
                    self.X_train_mean_var_in_Y[c] = {
                        "mean": such_X_train.mean(axis=0),
                        'var': np.var(such_X_train) + smoothing
                    }
                    self.lable_Prob_in_Y[c] = float(len(Y[Y == c])) / len(Y)

                print("len of X_train_mean_var_in_Y:", len(self.X_train_mean_var_in_Y))
                # print("X_train_mean_var_in_Y:", self.X_train_mean_var_in_Y)
            else:
                self.X_train_mean_cov_in_Y = {}
                for c in labels:
                    such_X_train = X_train[Y == c]
                    self.X_train_mean_cov_in_Y[c] = {
                        "mean": such_X_train.mean(axis=0), 
                        'cov': np.cov(such_X_train.T) + np.eye(D)*smoothing
                    }
                    self.lable_Prob_in_Y[c] = float(len(Y[Y == c])) / len(Y)

                print("len of X_train_mean_cov_in_Y:", len(self.X_train_mean_cov_in_Y))
                # print("X_train_mean_cov_in_Y:", self.X_train_mean_cov_in_Y)
            print("len of X_train_Prob_in_Y:", len(self.lable_Prob_in_Y))
            # print("lable_Prob_in_Y:", self.lable_Prob_in_Y)
        else:
            pass

    def predict(self, X_test):
        N = X_test.shape[0]
        print("shape of X_test:", X_test.shape)
        Y_pred_P = np.zeros((N, self.lable_numbers))

        if self.naive is True:
            for c, g in self.X_train_mean_var_in_Y.items():
                mean, var = g["mean"], g["var"]    
                Y_pred_P[:, c] = mvn.logpdf(X_test, mean=mean, cov=var) + np.log(self.lable_Prob_in_Y[c])
                # print("c:{0}, g:{1}".format(c, g))
        else:
            for c, g in self.X_train_mean_cov_in_Y.items():       
                mean, cov = g["mean"], g["cov"]                
                Y_pred_P[:, c] = mvn.logpdf(X_test, mean=mean, cov=cov) + np.log(self.lable_Prob_in_Y[c])
                # print("c:{0}, g:{1}".format(c, g))

        # print("Y_pred_P:{0}".format(Y_pred_P))
        Y_pred = np.argmax(Y_pred_P, axis=1)
        return Y_pred
