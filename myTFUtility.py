import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
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


def filter_summary(V, weight_shape):
    ix = weight_shape[0]
    iy = weight_shape[1]
    cx, cy = 8, 8
    V_T = tf.transpose(V, (3, 0, 1, 2))
    tf.summary.image("filters", V_T, max_outputs=64)


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

def conv_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0,
                                        dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,
                                         dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out],
                            initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out],
                            initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2],
                                            name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
                            
    mean, var = control_flow_ops.cond(phase_train,
                                      mean_var_with_update, 
                                      lambda: (ema_mean, ema_var))
    normed = tf.nn.batch_norm_with_global_normalization(x,
                                                        mean, 
                                                        var, 
                                                        beta, 
                                                        gamma, 
                                                        1e-3, 
                                                        True)
    return normed


def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0,
                                        dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0,
                                         dtype=tf.float32)
    beta = tf.get_variable("beta", [n_out],
                            initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out],
                            initializer=gamma_init)
    batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean),tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,
                                      mean_var_with_update, 
                                      lambda: (ema_mean, ema_var))

    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(x_r,
                                                        mean, 
                                                        var, 
                                                        beta, 
                                                        gamma, 
                                                        1e-3, 
                                                        True)
    return tf.reshape(normed, [-1, n_out])

def layer(inputs, weight_shape, bias_shape, activation=None):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    Z = tf.add(tf.matmul(inputs, W), b)

    if activation is None:
        outputs = Z
    else:
        outputs = activation(Z)

    return outputs

def layer_v2(inputs, weight_shape, bias_shape, phase_train, activation=None, visualize=False):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    Z = tf.add(tf.matmul(inputs, W), b)

    Z_nb = layer_batch_norm(Z, weight_shape[1], phase_train)

    if activation is None:
        outputs = Z_nb
    else:
        outputs = activation(Z_nb)

    return outputs

def conv2d(inputs, weight_shape, bias_shape, 
            padding_mode='VALID', activation=None, visualize=False):
    w_in = weight_shape[0] * weight_shape[1] * weight_shape[2]
    
    weight_init = tf.random_normal_initializer(stddev=
                                                (2.0/w_in)**0.5)
    bias_init = tf.constant_initializer(value=0)
    
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)

    if visualize:
        filter_summary(W, weight_shape)

    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    conv_out = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1],
                            padding=padding_mode)
    Z = tf.add(conv_out, b)
    
    if activation is None:
        outputs = Z
    else:
        outputs = activation(Z)
    return outputs

def conv2d_v2(inputs, weight_shape, bias_shape, 
            phase_train, padding_mode='VALID', 
            activation=None, visualize=False):
    w_in = weight_shape[0] * weight_shape[1] * weight_shape[2]
    
    weight_init = tf.random_normal_initializer(stddev=
                                                (2.0/w_in)**0.5)
    bias_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)

    if visualize:
        filter_summary(W, weight_shape)

    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    conv_out = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1],
                            padding=padding_mode)
    Z = tf.add(conv_out, b)

    Z_bn = conv_batch_norm(Z, weight_shape[3], phase_train)
    
    if activation is None:
        outputs = Z_bn
    else:
        outputs = activation(Z_bn)
    return outputs

def max_pool(inputs, k=2, padding_mode='VALID'):
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1],
                            strides=[1, k, k, 1], padding=padding_mode)


