import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

import os
import sys
cur_path = os.path.dirname(__file__)
rel_path = "..\\"
abs_file_path = os.path.join(cur_path, rel_path)
sys.path.insert(0, abs_file_path)
from myTFUtility import *


data_set = 'mnist'

if __name__ == '__main__':

    if data_set=='mnist':
        
        from tensorflow.examples.tutorials.mnist import input_data

        dataset = input_data.read_data_sets("MNIST_data", one_hot=True)        

        print('Train images:', dataset.train.images.shape)
        print('Train lables:', dataset.train.labels.shape)
        print('validation images:', dataset.validation.images.shape)
        print('validation lables:', dataset.validation.labels.shape)
        print('Test images:', dataset.test.images.shape)
        print('Test lables:', dataset.test.labels.shape)

        X_train = dataset.train.images
        Y_train = dataset.train.labels
        X_test = dataset.test.images
        Y_test = dataset.test.labels
        X_valid = dataset.validation.images
        Y_valid = dataset.validation.labels
        
        """
        cur_path = os.path.dirname(__file__)
        rel_path = ".\\..\Dataset\\mnist_scaled.npz"
        abs_file_path = os.path.join(cur_path, rel_path)
        mnist = np.load(abs_file_path)

        X_data, y_data, X_test, Y_test = [mnist[f] for f in ['X_train', 'y_train',
                                                     'X_test', 'y_test']]
        X_train, Y_train = X_data[:50000, :], y_data[:50000]
        X_valid, Y_valid = X_data[50000:, :], y_data[50000:]

        mean_vals = np.mean(X_train, axis=0)
        std_val = np.std(X_train)
        X_train = (X_train - mean_vals)/std_val
        X_valid = (X_valid - mean_vals)/std_val
        X_test = (X_test - mean_vals)/std_val

        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
        Y_valid = np_utils.to_categorical(Y_valid, 10)
        """

        # Parameters
        learning_rate = 0.001
        training_epochs = 30
        batch_size = 100
        nodes_input = 784
        nodes_fc = 1024
        nodes_output = 10
        dropout_rate = 0.5
        keep_prob = 1 - dropout_rate
        activation_h = tf.nn.relu
        # activation_output= tf.nn.softmax

        
        def inference(highlevelAPI=True):
            
            x = tf.placeholder(tf.float32, [None, nodes_input])
            y = tf.placeholder(tf.float32, [None, nodes_output])

            # reshape x to a 4D tensor:
            ##  [batchsize, width, height, 1]
            x_image = tf.reshape(x, shape=[-1, 28, 28, 1],
                                name='x_2dimages')

            is_train = tf.placeholder(tf.bool,
                                shape=(),
                                name='is_train')
            
            if highlevelAPI is True:
                # Placeholders for X and y:

                with tf.variable_scope("layer_conv1") as scope:
                    print("Build 1st conv layer:")
                    # Convolutuion 
                    output_conv1 = tf.layers.conv2d(x_image,
                                                kernel_size=(5, 5),
                                                filters=32,
                                                padding='VALID,
                                                activation=activation_h)
                    # MaxPooling
                    output_pool_conv1 = tf.layers.max_pooling2d(output_conv1,
                                                            pool_size=(2, 2),
                                                            strides=(2, 2))
                with tf.variable_scope("layer_conv2") as scope:
                    print("Build 2nd conv layer:")
                    # Convolutuion
                    output_conv2 = tf.layers.conv2d(output_pool_conv1, 
                                                kernel_size=(5, 5),
                                                filters=64,
                                                padding='VALID',
                                                activation=activation_h)
                    # MaxPooling
                    output_pool_conv2 = tf.layers.max_pooling2d(output_conv2,
                                                            pool_size=(2, 2),
                                                            strides=(2, 2))
                with tf.variable_scope("layer_fc") as scope:
                    print("Build fc layer:")
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc = tf.reshape(output_pool_conv2,
                                        shape=[-1, flat_units])
                    output_fc = tf.layers.dense(output_flat_fc, nodes_fc,
                                                activation=activation_h)
                    # Dropout
                    output_drop_fc = tf.layers.dropout(output_fc,
                                            rate=dropout_rate,
                                            training=is_train)
                with tf.variable_scope("layer_output") as scope:
                    print("Build output layer:")
                    output = tf.layers.dense(output_drop_fc, nodes_output,
                                        activation=None)
            else:
                with tf.variable_scope("layer_conv1"):
                    output_conv1 = conv2d(x_image, [5, 5, 1, 32], [32],
                                          padding_mode='VALID')
                    output_pool_conv1 = max_pool(output_conv1, k=2)

                with tf.variable_scope("layer_conv2"):
                    output_conv2 = conv2d(output_pool_conv1, [5, 5, 32, 64], [64],
                                          padding_mode='VALID')
                    output_pool_conv2 = max_pool(output_conv2)
                
                with tf.variable_scope("layer_fc"):
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc = tf.reshape(output_pool_conv2, [-1, flat_units])
                    output_fc = layer(output_flat_fc, [flat_units, nodes_fc], [nodes_fc])
                    # apply dropout
                    output_drop_fc = tf.nn.dropout(output_fc, keep_prob)
                with tf.variable_scope("output"):
                    output = layer(output_drop_fc, [nodes_fc, nodes_output], [nodes_output]) 
                    
            return output, x, y, is_train


        def loss(output, y):
            
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=output,
                    labels=y),
                    name='cross_entropy_loss')
            return cost


        def training(cost, global_step=1):
            """
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)

            """
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                        .minimize(loss=cost, name='train_op')
            
            return train_op


        def evaluate(output, y):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy

    elif data_set=='XXX':
        pass


    g = tf.Graph()
    with g.as_default():        

        output, x, y, is_train = inference(highlevelAPI=False)

        cost = loss(output, y)

        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # train_op = training(cost, global_step)
        train_op = training(cost)

        eval_op = evaluate(output, y)

        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init_op)
    
        epoch_list = []
        loss_list = []
        accuracy_list = []
        display_step = 5
    
        for epoch in range(training_epochs):     
            # Loop over all batches
            batch_gen = batch_generator(X_train, Y_train, 
                                        batch_size=batch_size,
                                        shuffle=True)
            for i, (minibatch_x, minibatch_y) in enumerate(batch_gen):
                # Fit training using batch data
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y, is_train:True})

            loss_, accuracy_ = sess.run([cost, eval_op], 
                                    feed_dict={x: X_valid, y: Y_valid, is_train:False})
            epoch_list.append(epoch)
            loss_list.append(loss_)
            accuracy_list.append(accuracy_)

            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch+1), 
                    "cost =", "{:.9f}".format(loss_),
                    "accuracy =", accuracy_)

        print("Optimization Finished!")
        print("Test Accuracy:", sess.run(eval_op, feed_dict={x: X_test, y: Y_test, is_train:False}))

        fig = plt.gcf()
        fig.set_size_inches(4, 2)
        plt.plot(epoch_list, loss_list, label="loss")
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss'], loc='upper left')
        plt.show()

        plt.plot(epoch_list, accuracy_list, label="accuracy")
        fig.set_size_inches(4, 2)
        plt.ylim(0.8, 1)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
    