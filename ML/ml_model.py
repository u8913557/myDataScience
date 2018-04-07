
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


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
        X_test, y_test = X[test_idx, :], y[test_idx]

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

    def __init__(self, method=None, learning_rate=0.01, num_epochs=100, shuffle=None):
        self.method = method
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.w_ = []
        self.cost_ = []
        #print("ml_model: __init__")


    def fit(self, X_train, Y, standardize=False):

        # Dimensions, Features of X_Train
        self.N, self.D = X_train.shape
        print("X_train has {0} samples with {1} features".format(self.N, self.D))     

        self.Y = Y

        if(standardize==True):
            X_std = np.copy(X_train) 
            for i in range(self.D):
                X_std[:,i] = (X_train[:,i] - X_train[:,i].mean()) / X_train[:,i].std()

            self.X_train = np.copy(X_std)

        else:
            self.X_train = np.copy(X_train)

        if (self.shuffle==True):
            #init w_ as random numbers
            self.w_ = np.random.randn(1 + self.D, 1)

            #shuffle X_train, Y
            r = np.random.permutation(self.N)
            self.X_train = self.X_train[r]
            self.Y = self.Y[r]

        else:
            # init w_ as zeros with w0
            self.w_ = np.zeros((1 + self.D, 1))        

        # Add one bias term
        ones = np.ones((self.N, 1))
        self.X_train = np.concatenate((ones, self.X_train), axis=1)
        
        print("shape of X_train:", self.X_train.shape)
        print("shape of w_:", self.w_.shape)
        print("shape of Y:", self.Y.shape)

    def activation_fn(self, z, activation=None):

            if(activation==None):
                return z
            elif(activation=="sign"):
                return np.piecewise(z, [z < 0, z >= 0], [-1, 1])
            elif(activation=="step"):
                return np.piecewise(z, [z < 0, z >= 0], [0, 1])

    def net_input(self, X_data):
        # net input: w0x0 + w1x1... + wixi
        #print("net_input:")
        #print("shape of X_data:", X_data.shape)
        #print("shape of w_:", self.w_.shape)
        return np.dot(X_data, self.w_)

    def predict(self):
        pass

    def r2_evl(self, Y, Y_hat):
        d1 = Y - Y_hat
        d2 = Y - Y.mean()
        #print("Shape of d1:", d1.shape)
        #print("Shape of d2:", d2.shape)
        r2 = 1 - (d1.T.dot(d1) / d2.T.dot(d2))
        print("R2:", r2[0][0])

    def score(self, Y, Y_hat):
        print('Misclassified samples: %d' % (Y != Y_hat).sum())
        print('Accuracy: %.2f' % accuracy_score(Y, Y_hat))
        print('Score:', np.mean(Y_hat == Y))
    

class myPerceptron(ml_model):
  
    def __init__(self, method=None, learning_rate=0.01, num_epochs=100, shuffle=True, activation="sign"):
        #print("perceptron: __init__")  
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
            if (0):
                Y_hat = self.predict(self.X_train)
                error = self.Y - Y_hat                
                delta_w = self.learning_rate * np.dot(self.X_train.T, error)
                self.w_ += delta_w
                self.errors_.append(error.mean())
            else :
                errors = 0
                for xi, y in zip(self.X_train, Y):
                    y_hat = self.predict(xi)          
                    error = y - y_hat
                    delta_w = self.learning_rate * np.dot(xi.reshape(xi.shape[0], 1), error.reshape(error.shape[0],1))
                    self.w_ += delta_w
                    errors += int(error != 0.0)
                self.errors_.append(errors)
              
        print("final w:\n", self.w_, "\nepochs:", (epoch+1), "/", self.num_epochs)
        plt.plot(range(1, len(self.errors_) + 1),
                self.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of erros')
        plt.show()

    def predict(self, X_data, addBias=False, standardize=False):
        
        if(standardize==True):
            for i in range(self.D):
                X_data[:,i] = (X_data[:,i] - X_data[:,i].mean()) / X_data[:,i].std()

        # Add one bias term
        if(addBias==True):
            ones = np.ones((X_data.shape[0], 1))
            X_data = np.concatenate((ones, X_data), axis=1)
        
        z = self.net_input(X_data)
        #print("z:",z)
        Y_hat = self.activation_fn(z, self.activation)
        #print("Y_hat:",Y_hat)
        return Y_hat

class myAdaline(ml_model):
  
    def __init__(self, method=None, learning_rate=0.0001, num_epochs=200, shuffle=True):
        #print("perceptron: __init__")  
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
            output = self.activation_fn(z, activation=None)
            error = (self.Y - output)
            self.w_[1:] += self.learning_rate * np.dot(self.X_train[:, 1:].T, error) 
            self.w_[0] += self.learning_rate * error.sum() 
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)                      
              
        print("final w:\n", self.w_, "\nFinal cost:\n", cost, "\nepochs:\n", (epoch+1), "/", self.num_epochs)

        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Sum-squared-error')
        plt.title('Adaline - Learning rate: {0}'.format(self.learning_rate))
        plt.tight_layout()
        plt.show()

    def predict(self, X_data, addBias=False, standardize=False):
    
        if(standardize==True):
            for i in range(self.D):
                X_data[:,i] = (X_data[:,i] - X_data[:,i].mean()) / X_data[:,i].std()

        # Add one bias term
        if(addBias==True):
            ones = np.ones((X_data.shape[0], 1))
            X_data = np.concatenate((ones, X_data), axis=1)
        
        z = self.net_input(X_data)
        output = self.activation_fn(z, activation=None)
        Y_hat = self.activation_fn(output, activation="sign")
        return Y_hat


        




