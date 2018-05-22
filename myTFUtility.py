import tensorflow as tf
import os
import numpy as np

def save_model(saver, sess, epoch, path='.//model//'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Saving model in %s' % path)
    saver.save(sess, os.path.join(path,'cnn-model.ckpt'),
               global_step=epoch)

    
def load_model(saver, sess, path, epoch):
    print('Loading model from %s' % path)
    saver.restore(sess, os.path.join(
            path, 'cnn-model.ckpt-%d' % epoch))

def batch_generator(X, y, batch_size=64,
                        shuffle=False, random_seed=None):

    # print("shape of X:", X.shape)
    # print("shape of y:", y.shape)
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size])

def layer(inputs, weight_shape, bias_shape, acivation=None):
    init = tf.random_uniform_initializer(minval=-1,
                                         maxval=1)
    W = tf.get_variable("W", weight_shape, initializer=init)
    b = tf.get_variable("b", bias_shape, initializer=init)
    Z = tf.add(tf.matmul(inputs, W), b)

    if acivation is None:
        outputs = Z
    else:
        outputs = acivation(Z)

    return outputs

def conv2d(input, weight_shape, bias_shape, 
            padding_mode='VALID', acivation=None):
    w_in = weight_shape[0] * weight_shape[1] * weight_shape[2]

    
    weight_init = tf.random_normal_initializer(stddev=
                                                (2.0/w_in)**0.5)
    bias_init = tf.constant_initializer(value=0)
    
    """
    init = tf.random_uniform_initializer(minval=-1,
                                         maxval=1)
    """
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)

    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    conv_out = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1],
                            padding=padding_mode)
    Z = tf.add(conv_out, b)
    
    if acivation is None:
        outputs = Z
    else:
        outputs = acivation(Z)
    return outputs

def max_pool(input, k=2, padding_mode='VALID'):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1],
                            strides=[1, k, k, 1], padding=padding_mode)