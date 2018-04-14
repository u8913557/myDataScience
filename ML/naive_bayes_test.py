import ml_model
from myDataset import get_donut
from myDataset import get_Iris
from myDataset import get_Wine
from myDataset import get_MNIST
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # Donut test
    print("Donut test:")

    X, Y = get_donut()

    plt.scatter(
        X[Y == 1, 0], X[Y == 1, 1], color='red',
        marker='o', label='Y=1')
    plt.scatter(
        X[Y == 0, 0], X[Y == 0, 1], color='blue',
        marker='x', label='Y=0')
    plt.title("Donut test")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    nb_test = ml_model.myNaiveBayes()
    nb_test.fit(X, Y)

    Y_hat = nb_test.predict(X)
    nb_test.r2_evl(Y, Y_hat)
    nb_test.score(Y, Y_hat)

    plt.scatter(
        X[Y == 1, 0], X[Y == 1, 1], color='red',
        marker='^', label='Y=1')
    plt.scatter(
        X[Y == 0, 0], X[Y == 0, 1], color='blue',
        marker='x', label='Y=0')
    plt.scatter(
        X[Y_hat == 1, 0], X[Y_hat == 1, 1], color='pink',
        marker='+', label="Y_hat=1")
    plt.scatter(
        X[Y_hat == 0, 0], X[Y_hat == 0, 1], color='green',
        marker='>', label="Y_hat=0")

    plt.title("Donut Dataset Predict result")
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    del nb_test

    # Iris Dataset test
    print("Iris Dataset test:")
    X, Y, selected_features, selected_lables = get_Iris(multi=True)
    Y[Y == selected_lables[0]] = 0
    Y[Y == selected_lables[1]] = 1
    Y[Y == selected_lables[2]] = 2

    # Prepare Data for training / testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    nb_test = ml_model.myNaiveBayes()
    nb_test.fit(X_train, y_train)

    Y_hat = nb_test.predict(X_test)
    nb_test.r2_evl(y_test, Y_hat)
    nb_test.score(y_test, Y_hat)

    del nb_test

    # Wine Dataset test
    print("Wine Dataset test:")

    X, Y, selected_features, selected_lables = get_Wine(multi=True)
    for i in range(len(selected_lables)):
            Y[Y == selected_lables[i]] = i

    # Prepare Data for training / testing
    X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=0)
    nb_test = ml_model.myNaiveBayes()
    nb_test.fit(X_train, y_train)

    Y_hat = nb_test.predict(X_test)
    nb_test.r2_evl(y_test, Y_hat)
    nb_test.score(y_test, Y_hat)

    del nb_test

    # MNIST Dataset
    print("MNIST Dataset:")
    X, Y = get_MNIST(limit=20000)

    X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.3, random_state=0)

    nb_test = ml_model.myNaiveBayes()
    nb_test.fit(X_train, y_train)
    Y_hat = nb_test.predict(X_test)
    nb_test.r2_evl(y_test, Y_hat)
    nb_test.score(y_test, Y_hat)
    del nb_test

