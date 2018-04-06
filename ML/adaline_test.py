import ml_model
from myDataset import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
        """
        # Donut test
        print("Donut test:")

        X, Y = get_donut()

        Y[Y == 0] = -1
        plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
                marker='o', label='Y=1')
        plt.scatter(X[Y==-1, 0], X[Y==-1, 1], color='blue', \
                marker='x', label='Y=-1')
        plt.title("Donut test")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        
        Y = Y.reshape(Y.shape[0], 1)
        adaline_test = ml_model.myAdaline(num_epochs=15, learning_rate=0.01)
        adaline_test.fit(X, Y, standardize=True)

        Y_hat = adaline_test.predict(X, addBias=True)
        adaline_test.r2_evl(Y, Y_hat)
        adaline_test.score(Y, Y_hat)

        plt.scatter(X[Y[:,0]==1, 0], X[Y[:,0]==1, 1], color='red', \
                marker='^', label='Y=1')
        plt.scatter(X[Y[:,0]==-1, 0], X[Y[:,0]==-1, 1], color='blue', \
                marker='x', label='Y=-1')
        plt.scatter(X[Y_hat[:, 0]==1, 0], X[Y_hat[:, 0]==1, 1], color='pink', \
                marker='+', label="Y_hat=1")
        plt.scatter(X[Y_hat[:, 0]==-1, 0], X[Y_hat[:, 0]==-1, 1], color='green', \
                marker='>', label="Y_hat=-1")
        plt.title("Donut Dataset Predict result")
        plt.xlabel("X0")
        plt.ylabel("X1")
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        del adaline_test
        """

        # Iris Dataset test
        # print("Iris Dataset test:")
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
        adaline_test = ml_model.myAdaline(num_epochs=300, learning_rate=0.01, shuffle=True)
        adaline_test.fit(X_train, y_train, standardize=True)

        Y_hat = adaline_test.predict(X_test, addBias=True, standardize=True)
        adaline_test.r2_evl(y_test, Y_hat)
        adaline_test.score(y_test, Y_hat) 
 
        
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
        
        del adaline_test

        
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
        adaline_test = ml_model.myAdaline(num_epochs=10, learning_rate=0.01)
        adaline_test.fit(X_train, y_train, standardize=True)

        plot_cost(adaline_test.cost_, adaline_test.learning_rate)
        
        Y_hat = adaline_test.predict(X_test, addBias=True, standardize=True)
        adaline_test.r2_evl(y_test, Y_hat)
        adaline_test.score(y_test, Y_hat)

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

        del adaline_test
        