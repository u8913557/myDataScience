from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve
from sklearn.naive_bayes import GaussianNB

import os
import sys
cur_path = os.path.dirname(__file__)
rel_path = "..\\"
abs_file_path = os.path.join(cur_path, rel_path)
sys.path.insert(0, abs_file_path)
from myUtility import plot_decision_regions


if __name__ == '__main__':

    # Donut
    print("Domut Test:")
    X, Y = datasets.make_circles(n_samples=200)
    # print("X:", X)
    # print("Y:", Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1,
                                                        stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    gnb = GaussianNB()
    gnb.fit(X_train_std, Y_train)
    Y_hat = gnb.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat,
                                                          normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % gnb.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=gnb)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Logistic)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = gnb
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    del gnb

    # Iris
    print("Iris Test:")
    iris_dataset = datasets.load_iris()
    X = iris_dataset.data
    indice = sorted(np.random.choice(X.shape[1], 2, replace=False))
    X = X[:, indice]
    # print("indice:", indice)
    # print("X:", X)
    Y = iris_dataset.target
    # print("Y:", Y)
    # print("Class lables:", np.unique(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1,
                                                        stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    gnb = GaussianNB()
    gnb.fit(X_train_std, Y_train)
    Y_hat = gnb.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat,
                                                          normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % gnb.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=gnb)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = gnb
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    del gnb

    # Wine
    print("Wine Test:")
    wine_dataset = datasets.load_wine()
    X = wine_dataset.data
    indice = sorted(np.random.choice(X.shape[1], 2, replace=False))
    X = X[:, indice]
    # print("X:", X)
    Y = wine_dataset.target
    # print("Y:", Y)
    # print("Class lables:", np.unique(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3,
                                                        random_state=1,
                                                        stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    gnb = GaussianNB()
    gnb.fit(X_train_std, Y_train)
    Y_hat = gnb.predict(X_test_std)
    print('Good classified samples: %d' % (accuracy_score(Y_test, Y_hat,
                                                          normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % gnb.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=gnb)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Naive Bayes)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = gnb
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    del gnb
