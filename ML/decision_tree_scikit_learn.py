from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score
from ml_model import plot_decision_regions
import matplotlib.pyplot as plt
from scikitplot.estimators import plot_learning_curve
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data


if __name__ == '__main__':
    # Donut
    print("Domut Test:")
    X, Y = datasets.make_circles(n_samples=200)
    # print("X:", X)
    # print("Y:", Y)

    X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train_std, Y_train)
    Y_hat = dt.predict(X_test_std)
    print('Good classified samples: %d' %
          (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % dt.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=dt)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Decision Tree)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = dt
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    dot_data = export_graphviz(dt,
                               filled=True,
                               rounded=True,
                               class_names=['0', '1'],
                               feature_names=['X0', 'X1'],
                               out_file=None)

    graph = graph_from_dot_data(dot_data)
    graph.write_png('Domut.png')

    del dt

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
    print("Class lables:", np.unique(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1, stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train_std, Y_train)
    Y_hat = dt.predict(X_test_std)
    print('Good classified samples: %d' % (
        accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % dt.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=dt)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Decision Tree)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = dt
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    dot_data = export_graphviz(dt,
                               filled=True,
                               rounded=True,
                               class_names=['0', '1', '2'],
                               feature_names=['X0', 'X1'],
                               out_file=None)

    graph = graph_from_dot_data(dot_data)
    graph.write_png('iris.png')

    del dt

    # Wine
    print("Wine Test:")
    wine_dataset = datasets.load_wine()
    X = wine_dataset.data
    indice = sorted(np.random.choice(X.shape[1], 2, replace=False))
    X = X[:, indice]
    # print("X:", X)
    Y = wine_dataset.target
    # print("Y:", Y)
    print("Class lables:", np.unique(Y))

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1, stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train_std, Y_train)
    Y_hat = dt.predict(X_test_std)
    print('Good classified samples: %d' %
          (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % dt.score(X_test_std, Y_test))

    plot_decision_regions(X=X_test_std, y=Y_test, classifier=dt)
    plt.xlabel('X0 [standardized]')
    plt.ylabel('X1 [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    title = "Learning Curves (Decision Tree)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = dt
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    dot_data = export_graphviz(dt,
                               filled=True,
                               rounded=True,
                               class_names=['0', '1', '2'],
                               feature_names=['X0', 'X1'],
                               out_file=None)

    graph = graph_from_dot_data(dot_data)
    graph.write_png('wine.png')

    del dt

    # MNIST
    print("MNIST Test:")
    digits_dataset = datasets.load_digits()
    X = digits_dataset.data
    Y = digits_dataset.target
    # print(X)
    # print(X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=1, stratify=Y)

    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train_std, Y_train)
    Y_hat = dt.predict(X_test_std)
    print('Good classified samples: %d' %
          (accuracy_score(Y_test, Y_hat, normalize=False)))
    print('Misclassified samples: %d' % (Y_test != Y_hat).sum())
    print('Accuracy: %.2f' % accuracy_score(Y_test, Y_hat))
    print('Accuracy: %.2f' % dt.score(X_test_std, Y_test))

    title = "Learning Curves (Decision Tree)"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = dt
    plot_learning_curve(estimator, X, Y, title, cv=cv, n_jobs=4)
    plt.show()

    dot_data = export_graphviz(dt,
                               filled=True,
                               rounded=True,
                               class_names=['0', '1', '2',
                                            '3', '4', '5',
                                            '6', '7', '8',
                                            '9'],
                               out_file=None)

    graph = graph_from_dot_data(dot_data)
    graph.write_png('MNIST.png')

    del dt
