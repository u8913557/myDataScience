from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from ml_model import plot_decision_regions
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve


if __name__ == '__main__':

    #Donut
    print("Domut Test:")
    X, Y = datasets.make_circles(n_samples=200)
    #print("X:", X)
    #print("Y:", Y)
    sc  = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    #print("X_std:", X_std)

    X_train_std, X_test_std, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.3, random_state=1, stratify=Y)

    ppn = Perceptron(max_iter=30, eta0=0.01, verbose=False)
    ppn.fit(X_train_std, Y_train)
    Y_hat = ppn.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % ppn.score(X_test_std, Y_test))

    plot_decision_regions(X=X_std, y=Y, classifier=ppn)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Perceptron)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = ppn
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    del ppn

    #Iris
    print("Iris Test:")
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data
    indice = sorted(np.random.choice(X.shape[1], 2, replace=False))
    X = X[:, indice]
    #print("indice:", indice)
    #print("X:", X)
    Y = iris_dataset.target
    #print("Y:", Y)
    #print("Class lables:", np.unique(Y))

    sc  = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    #print("X_std:", X_std)

    X_train_std, X_test_std, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.3, random_state=1, stratify=Y)

    ppn = Perceptron(max_iter=30, eta0=0.01, verbose=False)
    ppn.fit(X_train_std, Y_train)
    Y_hat = ppn.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % ppn.score(X_test_std, Y_test))

    plot_decision_regions(X=X_std, y=Y, classifier=ppn)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Perceptron)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = ppn
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    del ppn

    #Wine
    print("Wine Test:")
    wine_dataset = datasets.load_wine()
    X = wine_dataset.data
    indice = sorted(np.random.choice(X.shape[1], 2, replace=False))
    X = X[:, indice]
    #print("X:", X)
    Y = wine_dataset.target
    #print("Y:", Y)
    #print("Class lables:", np.unique(Y))

    sc  = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    #print("X_std:", X_std)

    X_train_std, X_test_std, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.3, random_state=1, stratify=Y)

    ppn = Perceptron(max_iter=30, eta0=0.01, verbose=False)
    ppn.fit(X_train_std, Y_train)
    Y_hat = ppn.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % ppn.score(X_test_std, Y_test))

    plot_decision_regions(X=X_std, y=Y, classifier=ppn)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Perceptron)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = ppn
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()


    del ppn