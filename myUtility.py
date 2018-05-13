
import matplotlib.pyplot as plt
import numpy as np

def plot_images_labels_prediction(images, labels,
                                  prediction, prediction_prob, 
                                  label_dict, idx, num=10):    
    fig = plt.gcf()
    fig.set_size_inches(14, 18)
    if num > 25: num = 25
    for i in range(num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(images[idx])        
        title = 'Index ' + str(i) + ': ' + label_dict[labels[i]]
        prob_text = ''
        if len(prediction) > 0:
            title += ' => ' + label_dict[prediction[i]]
            for j in range(len(label_dict)):
                prob_text += label_dict[j] + ' Prob.:%1.3f'%(prediction_prob[i][j]) + '\n'

        prob_text += '\n'
        ax.set_title(title, fontsize=10)
        ax.text(0, 120, prob_text, ha='left', wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    #plt.tight_layout()
    #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    plt.show()


# Show Keras train history
def show_train_history(history, valid_data_rate=0):
    # summarize history for accuracy
    plt.plot(history.history['acc'])

    if valid_data_rate != 0:
        plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])

    if valid_data_rate != 0:
        plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    from matplotlib.colors import ListedColormap

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v', '>', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'green', 'pink')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        # X_test, y_test = X[test_idx, :], y[test_idx]
        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100,
                        label='test set')


def conv2d(X, W, p=(0, 0), s=(1, 1)):
    W_rot = np.array(W)[::-1, ::-1]
    X_orig = np.array(X)
    n1 = X_orig.shape[0] + 2*p[0]
    n2 = X_orig.shape[1] + 2*p[1]
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]:p[0]+X_orig.shape[0],
             p[1]:p[1]+X_orig.shape[1]] = X_orig

    res = []
    for i in range(0, int((X_padded.shape[0] - \
        W_rot.shape[0])/s[0])+1, s[0]):
        
        res.append([])
        for j in range(0, int((X_padded.shape[1] - \
                       W_rot.shape[1])/s[1])+1, s[1]):

            X_sub = X_padded[i:i+W_rot.shape[0],
                             j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub * W_rot))

    return(np.array(res))