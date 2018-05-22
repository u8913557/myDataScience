import tensorflow as tf
import matplotlib.pyplot as plt

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

        # Parameters
        learning_rate = 0.001
        training_epochs = 100
        batch_size = 100

        nodes_input = 784
        nodes_h1 = 1000
        nodes_h2 = 1000
        nodes_output = 10

        activation_h = tf.nn.relu
        # activation_output= tf.nn.softmax

        
        def inference():
            # Placeholders for X and y:
            x = tf.placeholder(tf.float32, [None, nodes_input])
            y = tf.placeholder(tf.float32, [None, nodes_output])

            with tf.variable_scope("layer_h1") as scope:
                output_h1 = layer(inputs=x, weight_shape=[nodes_input, nodes_h1], 
                                  bias_shape=[nodes_h1], acivation=activation_h)
            with tf.variable_scope("layer_h2") as scope:
                output_h2 = layer(inputs=output_h1, weight_shape=[nodes_h1, nodes_h2], 
                                  bias_shape=[nodes_h2], acivation=activation_h)
            with tf.variable_scope("layer_output") as scope:
                output = layer(inputs=output_h2, weight_shape=[nodes_h2, nodes_output], 
                               bias_shape=[nodes_output], acivation=None)
            
            return output, x, y

        def loss(output, y):
            
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=output,
                    labels=y))
            return cost


        def training(cost, global_step=1):
            """
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)

            """
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate) \
                        .minimize(loss=cost)
            
            return train_op


        def evaluate(output, y):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return accuracy

    elif data_set=='XXX':
        pass


    g = tf.Graph()
    with g.as_default():        

        output, x, y = inference()

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
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})

            loss_, accuracy_ = sess.run([cost, eval_op], 
                                    feed_dict={x: X_valid, y: Y_valid})
            epoch_list.append(epoch)
            loss_list.append(loss_)
            accuracy_list.append(accuracy_)

            if epoch % display_step == 0:
                print("Epoch:", '%02d' % (epoch+1), 
                    "cost =", "{:.9f}".format(loss_),
                    "accuracy =", accuracy_)

        print("Optimization Finished!")
        print("Test Accuracy:", sess.run(eval_op, feed_dict={x: X_test, y: Y_test}))

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

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('myDataScience//log//MLP', sess.graph)
    