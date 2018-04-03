
import numpy as np

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

    def __init__(self, method=None, learning_rate=0.01, num_epochs=100, shuffle=None, activation=None):
        self.method = method
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.activation = activation
        self.w_ = []
        self.costs = []    
        #print("ml_model: __init__")


    def fit(self):
        pass        

    def activation_fn(self, z):
            if(self.activation=="step"):
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
        return r2

    def score(self, Y, Y_hat):
        return np.mean(Y_hat == Y)
    

class myPerceptron(ml_model):
  
    def __init__(self, method=None, eta=0.01, num_epochs=100, shuffle=True, activation='step'):
        #print("perceptron: __init__")  
        super().__init__(method, eta, num_epochs, shuffle, activation)          

    """
    Parameters
    ------------
    X_train : array, float
        Dataset for traninig
    Y : array, float
        "True" Y
    """
    def fit(self, X_train, Y):
        # print("perceptron: train")
        #super().train()

        # Dimensions, Features of X_Train
        N, D = X_train.shape

        if (self.shuffle==True):
            #init w_ as random numbers
            self.w_ = np.random.randn(1 + D, 1)

            #shuffle X_train, Y
            r = np.random.permutation(N)
            X_train = X_train[r]
            Y = Y[r]
        else:
            # init w_ as zeros with w0
            self.w_ = np.zeros(1 + D, 1)        

        # Add one bias term
        ones = np.ones((N, 1))
        X_train = np.concatenate((ones, X_train), axis=1)
        
        #print("shape of X_train:", X_train.shape)
        #print("shape of w_:", self.w_.shape)
        #print("shape of Y:", Y.shape)
        
        error=np.zeros(self.w_.shape)
        for epoch in range(self.num_epochs):
            Y_hat = self.predict(X_train)
            error = Y - Y_hat
                
            if(np.count_nonzero(error, axis=0)==0):
                print("Finish Training at {0}-th run".format(epoch+1))
                break
                
            delta_w = self.learning_rate * np.dot(X_train.T, error)
            self.w_ += delta_w
            #self.errors_.append(error)
              
        print("final w_:\n", self.w_, "\nepochs:", (epoch+1), "/", self.num_epochs)

    def net_input(self, X_data):
        # net input: w0x0 + w1x1... + wixi
        #print("net_input:")
        #print("shape of X_data:", X_data.shape)
        #print("shape of w_:", self.w_.shape)
        return np.dot(X_data, self.w_)

    def predict(self, z):
        if (self.activation):
            return self.activation_fn(self.net_input(z))
        else:      
            return input(z)

        




