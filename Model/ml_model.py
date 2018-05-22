
import numpy as np
# from sklearn.metrics import accuracy_score
from sortedcontainers import SortedList
import math
import matplotlib.pyplot as plt


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
            elif(activation == "tanh"):
                return (1.0 - np.exp(-2*z))/(1.0 + np.exp(-2*z))

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
            # print('errors:', errors)

        print("final w:\n", self.w_, "\nepochs:",
              (epoch+1), "/", self.num_epochs)
        plt.plot(range(1, len(self.errors_) + 1),
                 self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of erros')
        plt.show()

    def predict(self, X_test, standardize=False):
        # print("shape of X_test:", X_test.shape)
        if(standardize is True):
            for i in range(self.D):
                X_test[:, i] = \
                 (X_test[:, i] - X_test[:, i].mean()) / X_test[:, i].std()

        z = self.net_input(X_test)
        # print("z:",z)
        Y_hat = super().activation_fn(z, self.activation)
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
        output = super().activation_fn(self.net_input(xi), activation=None)
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
                output = super().activation_fn(z, activation=None)
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
        # output = super().activation_fn(z, activation=None)
        Y_hat = super().activation_fn(z, activation="sign")
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
            output = super().activation_fn(z, activation='sigmoid')
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
        # output = super().activation_fn(z, activation='sigmoid')
        Y_hat = super().activation_fn(z, activation="step")
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

        from scipy.stats import multivariate_normal as mvn
        
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


class myMLP(ml_model):

    def activation_derivative(self, z, activation=None):

            if(activation is None):
                return 0
            elif(activation == "sigmoid"):
                return (super().activation_fn(z, "sigmoid")*(1-super().activation_fn(z, "sigmoid")))
            elif(activation == "tanh"):
                return (1 + super().activation_fn(z, "tanh"))*(1 - super().activation_fn(z, "tanh"))

    def __init__(self, net_arch=[2, 3, 2], activation_h='tanh', l2=0.,
                 num_epochs=100, learning_rate=0.001, 
                 shuffle=True, minibatch_size=1):
        self.activation_h = activation_h
        self.layers = len(net_arch)
        self.net_arch = net_arch
        self.l2 = l2
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.shuffle = shuffle

    def _forward(self, X):
        """Compute forward propagation step"""

        A_in = [X]

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        # [n_samples, n_hidden] dot [i layer n_hidden, i+1 n_hidden]
        # print("shape of X:", X.shape)
        for i in range(len(self.weights)-1):
            print("shape of A_in[{0}]:{1}".format(i, A_in[i].shape))
            print("shape of weights({0}):{1}".format(i, self.weights[i].shape))
            z_h = np.dot(A_in[i], self.weights[i])
            a_h = self.activation_fn(z_h, self.activation_h)
            print("shape of a_h({0}):{1}".format(i, a_h.shape))
            # add the bias for the next layer
            ones = np.ones((1, X.shape[0]))
            a_h = np.concatenate((ones.T, a_h), axis=1)
            print("Shape of a_h with bias:", a_h.shape)              
            A_in.append(a_h)
            print("Len of A_in:", len(A_in)) 

        # step 2: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]
        z_o = np.dot(A_in[-1], self.weights[-1])             
        a_o = self.activation_fn(z_o, self.activation_h)           

        return A_in, a_o

    def _compute_cost(self, y, y_hat):
        """Compute cost function.

        Parameters
        ----------
        y : array, shape = (n_samples, n_labels)
            true class labels.
        y_hat : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)

        Returns
        ---------
        cost : float
            Regularized cost

        """
        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))
        """
        L2_term = 0

        y = y.reshape(y_hat.shape[0], y_hat.shape[1])
        print("shape of y:", y.shape)
        print("shape of y_hat:", y_hat.shape)
        term1 = -y * (np.log(y_hat))
        term2 = (1. - y) * np.log(1. - y_hat)
        print("shape of term1:", term1.shape)
        print("shape of term2:", term2.shape)
        cost = np.sum(term1 - term2) + L2_term
        return cost

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y, standardize=False):

        # Dimensions, Features of X_Train
        self.N, self.D = X_train.shape
        print("X_train has {0} samples with {1} features".format(self.N, self.D))

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

        ones = np.ones((1, self.X_train.shape[0]))        
        self.X_train = np.concatenate((ones.T, self.X_train), axis=1)
        print("Shape of X_train with bias:", self.X_train.shape)

        self.weights = []
        if (self.shuffle is True):
            # init weights as random numbers
            print("init weights as random numbers:")
            for layer in range(len(self.net_arch) - 1):
                w_ = np.random.randn(self.net_arch[layer] + 1, self.net_arch[layer+1])
                print("for layer {0} to {1}".format(layer, layer+1))
                print("shape of w_:", w_.shape)
                self.weights.append(w_)

            # shuffle X_train, Y
            # r = np.random.permutation(self.N)
            # self.X_train = self.X_train[r]
            # self.Y = self.Y[r]

        else:
            # init w_ as zeros with w0
            print("init weights as 0:")
            for layer in range(len(self.net_arch) - 1):  
                w_ = np.zeros(self.net_arch[layer] + 1, self.net_arch[layer+1])
                print("for layer {0} to {1}".format(layer, layer+1))
                print("shape of w_:", w_.shape)
                self.weights.append(w_)

        print("shape of X_train:", self.X_train.shape)
        print("len of weights:", len(self.weights))
        print("shape of Y:", self.Y.shape)

        # iterate over training epochs
        for i in range(self.num_epochs):
            
            indices = np.arange(X_train.shape[0])

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
                A_in, y_hat = self._forward(self.X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                print("shape of y[batch_idx]:", y[batch_idx].shape)
                print("shape of y_hat:", y_hat.shape)
                delta_out = y_hat - y[batch_idx].reshape(y_hat.shape[0], y_hat.shape[1])
                print("shape of delta_out:", delta_out.shape)

        # error for the output layer
        #self._compute_cost(Y, y_hat)


mlp = myMLP(net_arch=[2, 4, 1], minibatch_size=4)
X = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])

y = np.array([0, 1, 1, 0])

mlp.fit(X, y)