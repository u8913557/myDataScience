import ml_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # AND Gate test
    print("AND Gate test:")

    # Get Data
    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    # Data Analysis
    Y = np.array([0, 0, 0, 1])

    plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
            marker='o', label='Y=1')
    plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
            marker='x', label='Y=0')
    plt.title("AND Gate Test")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()    
    
    # Prepare Data for training
    Y[Y == 0] = -1
    Y = Y.reshape(Y.shape[0], 1)
    perceptron_test = ml_model.myPerceptron(num_epochs=200)
    perceptron_test.fit(X, Y)

    # Prepare Data for teating
    Y_hat = perceptron_test.predict(X, addBias=True)
    print("R2:", perceptron_test.r2_evl(Y, Y_hat))
    print("Score:", perceptron_test.score(Y, Y_hat))

    del perceptron_test

    # OR Gate test
    print("OR Gate test:")

    # Get Data
    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    Y = np.array([0, 1, 1, 1])

    # Data Analysis
    plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
            marker='o', label='Y=1')
    plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
            marker='x', label='Y=0')
    plt.title("OR Gate Test")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

    # Prepare Data for training
    Y[Y == 0] = -1
    Y = Y.reshape(Y.shape[0], 1)
    perceptron_test = ml_model.myPerceptron(num_epochs=200)
    perceptron_test.fit(X, Y)

    # Prepare Data for testing
    Y_hat = perceptron_test.predict(X, addBias=True)
    print("R2:", perceptron_test.r2_evl(Y, Y_hat))
    print("Score:", perceptron_test.score(Y, Y_hat))

    del perceptron_test

    # XOR Gate test
    print("XOR Gate test:")

    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    Y = np.array([0, 1, 1, 0])

    plt.scatter(X[Y==1, 0], X[Y==1, 1], color='red', \
            marker='o', label='Y=1')
    plt.scatter(X[Y==0, 0], X[Y==0, 1], color='blue', \
            marker='x', label='Y=0')
    plt.title("XOR Gate Test")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

    Y[Y == 0] = -1
    Y = Y.reshape(Y.shape[0], 1)
    perceptron_test = ml_model.myPerceptron(num_epochs=200)
    perceptron_test.fit(X, Y)

    Y_hat = perceptron_test.predict(X, addBias=True)
    print("R2:", perceptron_test.r2_evl(Y, Y_hat))
    print("Score:", perceptron_test.score(Y, Y_hat))

    del perceptron_test

    # Iris Dataset test
    """
    Three types of Iris:
    Iris-setosa, Iris-versicolor and Iris-virginica
    """
    print("Iris Dataset test:")

    # Get Data
    iris_df = pd.read_csv('iris.data', header=None)
    #print(iris_df.head(3))

    """ select setosa and versicolor """
    y_df = iris_df.iloc[0:100, 4].values
    y_df = np.where((y_df=="Iris-setosa"), 1, -1)
    y_df = y_df.reshape(y_df.shape[0], 1)

    """ extract sepal length and petal length """
    x_df = iris_df.iloc[0:100, [0, 2]].values
    
    # Data analysis
    plt.scatter(x_df[0:50, 0], x_df[0:50, 1], color='red', \
            marker='o', label='Iris-setosa')
    plt.scatter(x_df[50:100, 0], x_df[50:100, 1], color='blue', \
            marker='x', label='Iris-versicolor')
    plt.title("Iris Dataset Test")
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.show()

    # Prepare Data for training / testing
    X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=0)

    perceptron_test = ml_model.myPerceptron(num_epochs=200)
    perceptron_test.fit(X_train, y_train)
   
    Y_hat = perceptron_test.predict(X_test, addBias=True)
    print("R2:", perceptron_test.r2_evl(y_test, Y_hat))
    print("Score:", perceptron_test.score(y_test, Y_hat))












