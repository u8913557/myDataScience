
import numpy as np
import pandas as pd
import os

def get_AndGate():
    # Get Data
    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    Y = np.array([0, 0, 0, 1])

    return X, Y

def get_OrGate():
    # Get Data
    X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    Y = np.array([0, 1, 1, 1])
    return X, Y


def get_XorGate(multiData=False):
    if(multiData==False):
        X = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ])
        
        Y = np.array([0, 1, 1, 0])

    else:
        X = np.zeros((200, 2))
        X[:50] = np.random.random((50, 2)) / 2 + 0.5 # (0.5-1, 0.5-1)
        X[50:100] = np.random.random((50, 2)) / 2 # (0-0.5, 0-0.5)
        X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]]) # (0-0.5, 0.5-1)
        X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]]) # (0.5-1, 0-0.5)
        Y = np.array([0]*100 + [1]*100)

    return X, Y

def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))
    return X, Y


def get_Iris(multi=False):
    """
    Three types of Iris:
    Iris-setosa, Iris-versicolor and Iris-virginica
    """
    # Get Data
    cur_path = os.path.dirname(__file__)
    rel_path = "..\Dataset\iris.data"
    abs_file_path = os.path.join(cur_path, rel_path)

    iris_df = pd.read_csv(abs_file_path, header=None)
    iris_df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width',
                      'Species']
    #print(iris_df.head(3))
    lables = np.unique(iris_df['Species'])
    #print('Class labels', lables)

    if(multi==False):
        # Make new data frame:random choice 2 features + one y
        indice = np.random.choice(4, 2, replace=False)
        selected_features = iris_df.columns[indice]
        print("selected_features:", selected_features)
        
        indice = np.append(indice, [4])

        # Fixed index for debugging
        #indice = [0,2,4] 
        iris_df = iris_df[iris_df.columns[indice]]   
        #print("iris_df:", iris_df)     
        
        # Make new data frame:random choice 2 class lables from y (3 class lables)
        indice = np.random.choice(3, 2, replace=False)

        # Fixed index for debugging
        #indice = [0,2]
        selected_lables = lables[indice]
        print("selected_lables:", selected_lables)

        r = np.where((iris_df['Species']==selected_lables[0]) | (iris_df['Species']==selected_lables[1]))
        iris_df = iris_df.iloc[r]
        #print("iris_df:", iris_df)

        y = iris_df.iloc[:, 2].values
        y = np.where((y==selected_lables[0]), 1, -1)
        #print("y:", y)
        x = iris_df.iloc[:, [0, 1]].values
        #print("x:", x)

        return x, y, selected_features, selected_lables
    else:
        # To do: one hot for y ??
        y = iris_df.iloc[:, 4].values        
        #print("y:", y)
        x = iris_df.iloc[:, 0:3].values
        #print("x:", x)
        return x, y, iris_df.columns    

def get_Wine(multi=False):
    cur_path = os.path.dirname(__file__)
    rel_path = "..\Dataset\wine.data"
    abs_file_path = os.path.join(cur_path, rel_path)

    df_wine = pd.read_csv(abs_file_path, header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
    #print(df_wine.head(3))
    lables = np.unique(df_wine['Class label'])
    print('Class labels', lables)

    if(multi==False):
        # Make new data frame:random choice 13 features + one y
        indice = np.random.choice([1,13], 2, replace=False)
        selected_features = df_wine.columns[indice]
        print("selected_features:", selected_features)
        
        indice = np.append(indice, [0])  
        df_wine = df_wine[df_wine.columns[indice]]   
        #print("df_wine:", df_wine)

        # Make new data frame:random choice 2 class lables from y (3 class lables)
        indice = np.random.choice(3, 2, replace=False)
        selected_lables = lables[indice]
        print("selected_lables:", selected_lables)

        r = np.where((df_wine['Class label']==selected_lables[0]) | (df_wine['Class label']==selected_lables[1]))
        df_wine = df_wine.iloc[r]
        #print("df_wine:", df_wine)

        y = df_wine.iloc[:, 2].values
        y = np.where((y==selected_lables[0]), 1, -1)
        #print("y:", y)
        x = df_wine.iloc[:, [0, 1]].values
        #print("x:", x)

        return x, y, selected_features, selected_lables
    else:
        # To do: one hot for y ??
        y = df_wine.iloc[:, 0].values        
        #print("y:", y)
        x = df_wine.iloc[:, 1:].values
        #print("x:", x)
        return x, y, df_wine.columns 
