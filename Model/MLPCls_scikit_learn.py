from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Iris

iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = \
        train_test_split(X, Y, test_size=0.3, random_state=1)      

sc = StandardScaler()
sc.fit(X)
X_train_std = sc.transform(X_train)                              
X_test_std = sc.transform(X_test)

mlp = MLPClassifier(max_iter=200, random_state=1,
                    hidden_layer_sizes=(200, 100))
mlp.fit(X_train_std, Y_train)

Y_pred = mlp.predict(X_test_std)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(Y_test, Y_pred))
pred_prob = mlp.predict_proba(X_test_std)
print('Probility:{0}'.format(pred_prob))
