import numpy as np


class Perceptron_m(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, shuffle=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        if(self.shuffle==True):
            rgen = np.random.RandomState(1)
            self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        else:
            self.w_ = [0,0,0]

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        print("final w:\n", self.w_, "\nepochs:", (_+1), "/", self.n_iter)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value in each epoch.

    """
    
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_seed = 1, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_seed = random_seed
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        """init w_ as zeros"""
        #self.w_ = np.zeros(1 + X.shape[1])
        """init w_ as random numbers"""
        rgen = np.random.RandomState(self.random_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])        
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            """
            Please note that the "activation" method has no effect
            in the code since it is simply an identity function. We
            could write `output = self.net_input(X)` directly instead.
            The purpose of the activation is more conceptual, i.e.,  
            in the case of logistic regression, we could change it to
            a sigmoid function to implement a logistic regression classifier. """
            output = self.activation(X)
            errors = (y - output)
            #print("errors:", errors)
            """ self.eta * X.T.dot(errors) = self.eta * X.T.dot((y - output)) = delta of wj """
            self.w_[1:] += self.eta * X.T.dot(errors) 
            self.w_[0] += self.eta * errors.sum()
            """ # Cost function J(Wj)"""
            cost = (errors**2).sum() / 2.0 
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)