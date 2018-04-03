import ml_model
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # AND Gate
    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    Y = np.array([0, 0, 0, 1])

    plt.scatter(X[Y==1, 0]==1, X[Y==1, 1]==1, color='red', \
            marker='o', label='Y=1')
    plt.scatter(X[Y==0, 0]==1, X[Y==0, 1]==1, color='blue', \
            marker='x', label='Y=0')
    plt.show()    
    
    Y[Y == 0] = -1
    perceptron_test = ml_model.perceptron(num_epochs=200)
    perceptron_test.fit(X, Y)

    ones = np.ones((X.shape[0], 1))
    X_test = np.concatenate((ones, X), axis=1)
    Y_hat = perceptron_test.predict(X_test)
    print("R2:", perceptron_test.r2_evl(Y, Y_hat))
    print("Score:", perceptron_test.score(Y, Y_hat))





