
import numpy as np
from sklearn.metrics import accuracy_score

class ml_model(object):
    """
    Parameters
    ------------
    method : str
        which ML lib
    eta : float (default: 0.01)
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
        self.isShuffled = False
        #print("ml_model: __init__")


    def fit(self, X_train, Y, standardize=False):
        self.isShuffled=False        

        if(standardize==True):
            X_train = (X_train - X_train.mean()) / X_train.std()
            self.X_train = X_train
            self.Y = Y
        else:
            self.X_train = X_train
            self.Y = Y

        # Dimensions, Features of X_Train
        self.N, self.D = self.X_train.shape
        print("X_train has {0} samples with {1} features".format(self.N, self.D))

    def activation_fn(self, z, activation=None):
            if(activation=="step"):
                return np.piecewise(z, [z < 0, z >= 0], [-1, 1])
            else:
                return z

    def predict(self):
        pass

    def r2_evl(self, Y, Y_hat):
        d1 = Y - Y_hat
        d2 = Y - Y.mean()
        #print("Shape of d1:", d1.shape)
        #print("Shape of d2:", d2.shape)
        r2 = 1 - (d1.T.dot(d1) / d2.T.dot(d2))
        print("R2:", r2)

    def score(self, Y, Y_hat):
        print('Accuracy: %.2f' % accuracy_score(Y, Y_hat))
        print('Score:', np.mean(Y_hat == Y))
    

class myPerceptron(ml_model):
  
    def __init__(self, method=None, eta=0.01, num_epochs=100, shuffle=True):
        #print("perceptron: __init__")  
        super().__init__(method, eta, num_epochs, shuffle)          

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y, standardize=False):
        # print("perceptron: train")
        super().fit(X_train, Y, standardize)
        
        if (self.shuffle==True and self.isShuffled==False):
            #init w_ as random numbers
            self.w_ = np.random.randn(1 + self.D, 1)

            #shuffle X_train, Y
            r = np.random.permutation(self.N)
            self.X_train = self.X_train[r]
            self.Y = self.Y[r]

            self.isShuffled=True
        else:
            # init w_ as zeros with w0
            self.w_ = np.zeros(1 + self.D, 1)        

        # Add one bias term
        ones = np.ones((self.N, 1))
        self.X_train = np.concatenate((ones, self.X_train), axis=1)
        
        #print("shape of X_train:", self.X_train.shape)
        #print("shape of w_:", self.w_.shape)
        #print("shape of Y:", self.Y.shape)
        
        error=np.zeros(self.w_.shape)
        for epoch in range(self.num_epochs):
            Y_hat = self.predict(self.X_train)
            error = self.Y - Y_hat
                
            if(np.count_nonzero(error, axis=0)==0):
                print("Finish Training at {0}-th run".format(epoch+1))
                break
                
            delta_w = self.learning_rate * np.dot(self.X_train.T, error)
            self.w_ += delta_w
            #self.errors_.append(error)
              
        print("final w:\n", self.w_, "\nepochs:", (epoch+1), "/", self.num_epochs)

    def net_input(self, X_data):
        # net input: w0x0 + w1x1... + wixi
        #print("net_input:")
        #print("shape of X_data:", X_data.shape)
        #print("shape of w_:", self.w_.shape)
        return np.dot(X_data, self.w_)

    def predict(self, z, addBias=False, standardize=False,activation="step"):
        if(standardize==True):
            z = (z - z.mean()) / z.std()

        # Add one bias term
        if(addBias==True):
            ones = np.ones((z.shape[0], 1))
            z = np.concatenate((ones, z), axis=1)
        
        return self.activation_fn(self.net_input(z), activation)


        




