import ml_model
from myDataset import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
        # AND Gate test
        print("AND Gate test:")

        X, Y = get_AndGate()

        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label='Y=1')
        plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
                marker='x', label='Y=0')
        plt.title("AND Gate Test")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper middle')
        plt.tight_layout()
        plt.show()    
        
        # Prepare Data for training
        Y[Y == 0] = -1
        Y = Y.reshape(Y.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X, Y)

        # Prepare Data for teating
        Y_hat = perceptron_test.predict(X, addBias=True)
        perceptron_test.r2_evl(Y, Y_hat)
        perceptron_test.score(Y, Y_hat)
        del perceptron_test

        # OR Gate test
        print("OR Gate test:")

        X, Y = get_OrGate()

        # Data Analysis
        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label='Y=1')
        plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
                marker='x', label='Y=0')
        plt.title("OR Gate Test")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper middle')
        plt.tight_layout()
        plt.show()

        # Prepare Data for training
        Y[Y == 0] = -1
        Y = Y.reshape(Y.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X, Y)

        # Prepare Data for testing
        Y_hat = perceptron_test.predict(X, addBias=True)
        perceptron_test.r2_evl(Y, Y_hat)
        perceptron_test.score(Y, Y_hat)

        del perceptron_test

        # XOR Gate test
        print("XOR Gate test:")

        X, Y = get_XorGate()

        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label='Y=1')
        plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
                marker='x', label='Y=0')
        plt.title("XOR Gate Test")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper middle')
        plt.tight_layout()
        plt.show()

        Y[Y == 0] = -1
        Y = Y.reshape(Y.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X, Y)

        Y_hat = perceptron_test.predict(X, addBias=True)
        perceptron_test.r2_evl(Y, Y_hat)
        perceptron_test.score(Y, Y_hat)

        del perceptron_test

        # Donut test
        print("Donut test:")

        X, Y = get_donut()

        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label='Y=1')
        plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
                marker='x', label='Y=0')
        plt.title("Donut test")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        Y[Y == 0] = -1
        Y = Y.reshape(Y.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X, Y)

        Y_hat = perceptron_test.predict(X, addBias=True)
        perceptron_test.r2_evl(Y, Y_hat)
        perceptron_test.score(Y, Y_hat)

        plt.scatter(X[Y[:,0]==1, 0], X[Y[:,0]==1, 1], color='red', \
                marker='^', label='Y=1')
        plt.scatter(X[Y[:,0]==-1, 0], X[Y[:,0]==-1, 1], color='blue', \
                marker='x', label='Y=0')
        plt.scatter(X[Y_hat[:, 0]==1, 0], X[Y_hat[:, 0]==1, 1], color='pink', \
                marker='+', label="Y_hat=1")
        plt.scatter(X[Y_hat[:, 0]==-1, 0], X[Y_hat[:, 0]==-1, 1], color='green', \
                marker='>', label="Y_hat=0")
        plt.title("Donut Dataset Predict result")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        del perceptron_test

        # Iris Dataset test
        print("Iris Dataset test:")

        X, Y, selected_features, selected_lables = get_Iris()    
        
        # Data analysis
        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label=selected_lables[0])
        plt.scatter(X[Y==-1, 0], X[Y==-1, 1], color='blue', \
                marker='x', label=selected_lables[1])
        plt.title("Iris Dataset Test")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Prepare Data for training / testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X_train, y_train, standardize=True)

        Y_hat = perceptron_test.predict(X_test, addBias=True, standardize=True)
        perceptron_test.r2_evl(y_test, Y_hat)
        perceptron_test.score(y_test, Y_hat)

        plt.scatter(X_test[y_test[:, 0]==1, 0], X_test[y_test[:,0]==1, 1], color='red', \
                marker='^', label=selected_lables[0])
        plt.scatter(X_test[y_test[:, 0]==-1, 0], X_test[y_test[:, 0]==-1, 1], color='blue', \
                marker='x', label=selected_lables[1])
        plt.scatter(X_test[Y_hat[:, 0]==1, 0], X_test[Y_hat[:, 0]==1, 1], color='pink', \
                marker='+', label='Predict '+ selected_lables[0])
        plt.scatter(X_test[Y_hat[:, 0]==-1, 0], X_test[Y_hat[:, 0]==-1, 1], color='green', \
                marker='>', label='Predict '+ selected_lables[1])
        plt.title("Iris Dataset Predict result")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Wine Dataset test
        print("Wine Dataset test:")

        X, Y, selected_features, selected_lables = get_Wine()    
        # Data analysis
        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label=selected_lables[0])
        plt.scatter(X[Y==-1, 0], X[Y==-1, 1], color='blue', \
                marker='x', label=selected_lables[1])
        plt.title("Wine Dataset Test")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Prepare Data for training / testing
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        y_train = y_train.reshape(y_train.shape[0], 1)
        y_test = y_test.reshape(y_test.shape[0], 1)
        perceptron_test = ml_model.myPerceptron(num_epochs=200)
        perceptron_test.fit(X_train, y_train, standardize=True)
        
        Y_hat = perceptron_test.predict(X_test, addBias=True, standardize=True)
        perceptron_test.r2_evl(y_test, Y_hat)
        perceptron_test.score(y_test, Y_hat)

        plt.scatter(X_test[y_test[:, 0]==1, 0], X_test[y_test[:,0]==1, 1], color='red', \
                marker='^', label=selected_lables[0])
        plt.scatter(X_test[y_test[:, 0]==-1, 0], X_test[y_test[:, 0]==-1, 1], color='blue', \
                marker='x', label=selected_lables[1])
        plt.scatter(X_test[Y_hat[:, 0]==1, 0], X_test[Y_hat[:, 0]==1, 1], color='pink', \
                marker='+', label=selected_lables[0])
        plt.scatter(X_test[Y_hat[:, 0]==-1, 0], X_test[Y_hat[:, 0]==-1, 1], color='green', \
                marker='>', label=selected_lables[1])
        plt.title("Wine Dataset Predict result")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()











