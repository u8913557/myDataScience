import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

import time
import os
import sys
cur_path = os.path.dirname(__file__)
rel_path = "..\\"
abs_file_path = os.path.join(cur_path, rel_path)
sys.path.insert(0, abs_file_path)
from myTFUtility import *


data_set = 'mnist'
# data_set = 'cifar10'

isNB = True

if __name__ == '__main__':

    if data_set=='mnist':
        
        from tensorflow.examples.tutorials.mnist import input_data

        with tf.device('/cpu:0'):
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

            X_train_processed = X_train
            X_test_processed = X_test
            X_valid_processed = X_valid                  

        # Parameters
        learning_rate = 0.001
        training_epochs = 300
        batch_size = 128
        nodes_input = 784
        nodes_fc = 128
        nodes_output = 10
        dropout_rate = 0.2
        keep_rate = 1 - dropout_rate
        activation_h = tf.nn.relu
        # activation_output= tf.nn.softmax
        activation_output= tf.nn.relu

        
        def inference(isNB=False):
            
            x = tf.placeholder(tf.float32, [None, nodes_input])
            y = tf.placeholder(tf.int32, [None, nodes_output])

            # reshape x to a 4D tensor:
            ##  [batchsize, width, height, 1]
            x_image = tf.reshape(x, shape=[-1, 28, 28, 1],
                                name='x_2dimages')

            keep_prob = tf.placeholder(tf.float32)           
            
            if isNB:
                is_train = tf.placeholder(tf.bool) # training or testing

                with tf.variable_scope("layer_conv1"):
                    output_conv1 = conv2d_v2(inputs=x_image, weight_shape=[5, 5, 1, 16], 
                                             bias_shape=[16], padding_mode='SAME',
                                             phase_train=is_train,
                                             activation=activation_h)
                    output_pool_conv1 = max_pool(inputs=output_conv1, k=2, 
                                                 padding_mode='SAME')

                with tf.variable_scope("layer_conv2"):
                    output_conv2 = conv2d_v2(inputs=output_pool_conv1, weight_shape=[5, 5, 16, 36], 
                                             bias_shape=[36], padding_mode='SAME',
                                             phase_train=is_train,
                                             activation=activation_h)
                    output_pool_conv2 = max_pool(inputs=output_conv2, k=2, padding_mode='SAME')
                    
                with tf.variable_scope("layer_fc"):
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc = tf.reshape(output_pool_conv2, [-1, flat_units])
                    output_fc = layer_v2(inputs=output_flat_fc, weight_shape=[flat_units, nodes_fc], 
                                         bias_shape=[nodes_fc],
                                         phase_train=is_train,
                                         activation=activation_h)
                    # apply dropout
                    output_drop_fc = tf.nn.dropout(output_fc, keep_prob)
                
                with tf.variable_scope("output"):
                    output = layer(inputs=output_drop_fc, weight_shape=[nodes_fc, nodes_output], 
                                   bias_shape=[nodes_output], activation=activation_output)

                return output, x, y, keep_prob, is_train
                
            else:
                
                with tf.variable_scope("layer_conv1"):
                    output_conv1 = conv2d(inputs=x_image, weight_shape=[5, 5, 1, 16], 
                                          bias_shape=[16], padding_mode='SAME',
                                          activation=activation_h)
                    output_pool_conv1 = max_pool(inputs=output_conv1, k=2, 
                                                 padding_mode='SAME')

                with tf.variable_scope("layer_conv2"):
                    output_conv2 = conv2d(inputs=output_pool_conv1, weight_shape=[5, 5, 16, 36], 
                                          bias_shape=[36], padding_mode='SAME',
                                          activation=activation_h)
                    output_pool_conv2 = max_pool(inputs=output_conv2, k=2, padding_mode='SAME')
                    
                with tf.variable_scope("layer_fc"):
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc = tf.reshape(output_pool_conv2, [-1, flat_units])
                    output_fc = layer(inputs=output_flat_fc, weight_shape=[flat_units, nodes_fc], 
                                      bias_shape=[nodes_fc], activation=activation_h)
                    
                    # apply dropout
                    output_drop_fc = tf.nn.dropout(output_fc, keep_prob)
                    
                with tf.variable_scope("output"):
                    output = layer(inputs=output_drop_fc, weight_shape=[nodes_fc, nodes_output], 
                                   bias_shape=[nodes_output], activation=activation_output) 
                    
                return output, x, y, keep_prob


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

    elif data_set=='cifar10':    
        
        rel_path = "D:\\tmp\\cifar10"
        sys.path.insert(0, rel_path)
        import cifar10,cifar10_input

        data_dir = 'D:\\tmp\\cifar10_data\\cifar-10-batches-bin'

        cifar10.maybe_download_and_extract()
        
        # Parameters
        learning_rate = 0.001
        training_epochs = 1000
        batch_size = 128
        total_batch = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size)
        # nodes_input = 24, 24, 3
        nodes_fc1 = 384
        nodes_fc2 = 192
        nodes_output = 10
        dropout_rate = 0.5
        keep_rate = 1 - dropout_rate
        activation_h = tf.nn.relu
        activation_output= tf.nn.relu

        def inference(isNB=False):            
            
            x = tf.placeholder(tf.float32, [None, 24, 24, 3])
            y = tf.placeholder(tf.int32, [None])

            keep_prob = tf.placeholder(tf.float32)

            if isNB:
                is_train = tf.placeholder(tf.bool) # training or testing

                with tf.variable_scope("layer_conv1"):
                    output_conv1 = conv2d_v2(inputs=x, weight_shape=[5, 5, 3, 64], 
                                          bias_shape=[64], phase_train=is_train, padding_mode='SAME',
                                          activation=activation_h, visualize=True)
                    output_pool_conv1 = max_pool(inputs=output_conv1, k=2, 
                                                 padding_mode='SAME')

                with tf.variable_scope("layer_conv2"):
                    output_conv2 = conv2d_v2(inputs=output_pool_conv1, weight_shape=[5, 5, 64, 64], 
                                          bias_shape=[64], padding_mode='SAME',
                                          phase_train=is_train,
                                          activation=activation_h)
                    output_pool_conv2 = max_pool(inputs=output_conv2, k=2, padding_mode='SAME')
                    
                with tf.variable_scope("layer_fc1"):
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc1 = tf.reshape(output_pool_conv2, [-1, flat_units])
                    output_fc1 = layer_v2(inputs=output_flat_fc1, weight_shape=[flat_units, nodes_fc1], 
                                       bias_shape=[nodes_fc1],
                                       phase_train=is_train,
                                       activation=activation_h)
                    # apply dropout
                    output_drop_fc1 = tf.nn.dropout(output_fc1, keep_prob)
                
                with tf.variable_scope("layer_fc2"):
                    output_fc2 = layer_v2(inputs=output_drop_fc1, weight_shape=[nodes_fc1, nodes_fc2], 
                                       bias_shape=[nodes_fc2],
                                       phase_train=is_train,
                                       activation=activation_h)
                    # apply dropout
                    output_drop_fc2 = tf.nn.dropout(output_fc2, keep_prob)

                with tf.variable_scope("output"):
                    output = layer_v2(inputs=output_drop_fc2, weight_shape=[nodes_fc2, nodes_output], 
                                   bias_shape=[nodes_output],
                                   phase_train=is_train,
                                   activation=activation_output)
                
                return output, x, y, keep_prob, is_train

            else:
                
                with tf.variable_scope("layer_conv1"):
                    output_conv1 = conv2d(inputs=x, weight_shape=[5, 5, 3, 64], 
                                          bias_shape=[64], padding_mode='SAME',
                                          activation=activation_h, visualize=True)
                    output_pool_conv1 = max_pool(inputs=output_conv1, k=2, 
                                                 padding_mode='SAME')

                with tf.variable_scope("layer_conv2"):
                    output_conv2 = conv2d(inputs=output_pool_conv1, weight_shape=[5, 5, 64, 64], 
                                          bias_shape=[64], padding_mode='SAME',
                                          activation=activation_h)
                    output_pool_conv2 = max_pool(inputs=output_conv2, k=2, padding_mode='SAME')
                    
                with tf.variable_scope("layer_fc1"):
                    input_shape = output_pool_conv2.get_shape().as_list()
                    flat_units = np.prod(input_shape[1:])
                    output_flat_fc1 = tf.reshape(output_pool_conv2, [-1, flat_units])
                    output_fc1 = layer(inputs=output_flat_fc1, weight_shape=[flat_units, nodes_fc1], 
                                       bias_shape=[nodes_fc1], activation=activation_h)
                    # apply dropout
                    output_drop_fc1 = tf.nn.dropout(output_fc1, keep_prob)
                
                with tf.variable_scope("layer_fc2"):
                    output_fc2 = layer(inputs=output_drop_fc1, weight_shape=[nodes_fc1, nodes_fc2], 
                                       bias_shape=[nodes_fc2], activation=activation_h)
                    # apply dropout
                    output_drop_fc2 = tf.nn.dropout(output_fc2, keep_prob)

                with tf.variable_scope("output"):
                    output = layer(inputs=output_drop_fc2, weight_shape=[nodes_fc2, nodes_output], 
                                   bias_shape=[nodes_output], activation=activation_output)
                
            return output, x, y, keep_prob

        def loss(output, y):            
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.cast(y, tf.int64))    
            cost = tf.reduce_mean(xentropy)
            return cost


        def training(cost, global_step):
            tf.summary.scalar("cost", cost)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)
        
            return train_op


        def evaluate(output, y):
            correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), dtype=tf.int32), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("validation error", (1.0 - accuracy))
            return accuracy

    g = tf.Graph()
    with g.as_default():

        if data_set=='cifar10':
            
            with tf.device('/cpu:0'):
                X_train_processed, Y_train = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                                            batch_size=batch_size)

                X_test_processed, Y_test = cifar10_input.inputs(eval_data=False,
                                                                data_dir=data_dir,
                                                                batch_size=batch_size)
                
                X_valid_processed, Y_valid = cifar10_input.inputs(eval_data=True,
                                                                data_dir=data_dir,
                                                                batch_size=batch_size)
            
        if isNB:
            output, x, y, keep_prob, is_train = inference(isNB=True)
        else:
            output, x, y, keep_prob = inference(isNB=False)

        cost = loss(output, y)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = training(cost, global_step)

        eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        saver = tf.train.Saver()

        sess = tf.Session()

        init_op = tf.global_variables_initializer()

        sess.run(init_op)
    
        train_writer = tf.summary.FileWriter(cur_path + "\\log\\CNN", sess.graph)

        tf.train.start_queue_runners(sess=sess)
    
        epoch_list = []
        loss_list = []
        accuracy_list = []
        display_step = 5
        
        for epoch in range(training_epochs):     
            # Loop over all batches
            start_time = time.time()
            avg_cost = 0.

            if data_set=='cifar10':               
                for i in range(total_batch):
                    # Fit training using batch data
                    minibatch_x_train, minibatch_y_train = sess.run([X_train_processed, Y_train])
                    
                    if isNB:
                        _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x_train, 
                                                    y: minibatch_y_train,                                                           
                                                    keep_prob: 1,
                                                    is_train: True})
                    else:
                        _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x_train, 
                                                    y: minibatch_y_train,                                                           
                                                    keep_prob: keep_rate})
                    avg_cost += new_cost/total_batch
                    # print("Epoch %d, minibatch %d of %d. Cost = %0.4f." %(epoch, i, total_batch, new_cost))

                duration = time.time() - start_time
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

                    minibatch_x_valid, minibatch_y_valid = sess.run([X_valid_processed, Y_valid])

                    if isNB:
                        loss_, accuracy_ = sess.run([cost, eval_op], feed_dict={x: minibatch_x_valid, y: minibatch_y_valid, 
                                                        keep_prob: 1,
                                                        is_train: False})
                    else:
                        loss_, accuracy_ = sess.run([cost, eval_op], feed_dict={x: minibatch_x_valid, y: minibatch_y_valid, keep_prob: 1})

                    examples_per_sec = batch_size / duration
                    sec_per_epoch = float(duration)

                    print("Epoch:", '%02d' % (epoch+1), 
                        "cost =", "{:.9f}".format(loss_),
                        "Validation accuracy =", accuracy_,
                        "; %.1f samples/sec; %.3f sec/epoch" % (examples_per_sec, sec_per_epoch))

                    if isNB:
                        summary_str = sess.run(summary_op, feed_dict={x: minibatch_x_train, y: minibatch_y_train, 
                                                    keep_prob: 1,
                                                    is_train: False})
                    else:
                        summary_str = sess.run(summary_op, feed_dict={x: minibatch_x_train, y: minibatch_y_train, keep_prob: 1})
                    
                    train_writer.add_summary(summary_str, sess.run(global_step))

                    saver.save(sess, cur_path + "\\SaveModel\\CNN\\tf\\model-checkpoint", global_step=global_step)                  

            else:
                batch_gen = batch_generator(X_train_processed, Y_train, 
                                            shuffle=True)
                for i, (minibatch_x, minibatch_y) in enumerate(batch_gen):
                    # Fit training using batch data
                    if isNB:
                        _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x, y: minibatch_y, 
                                                    keep_prob: 1,
                                                    is_train: True})
                    else:
                        _, new_cost = sess.run([train_op, cost], feed_dict={x: minibatch_x, y: minibatch_y, keep_prob: keep_rate})
                    #print("Epoch %d, minibatch #%d. new_cost = %0.4f." %(epoch, i, new_cost))

                if isNB:
                    loss_, accuracy_ = sess.run([cost, eval_op], 
                                     feed_dict={x: X_valid_processed, y: Y_valid, 
                                                    keep_prob: 1,
                                                    is_train: False})
                else:
                    loss_, accuracy_ = sess.run([cost, eval_op], 
                                                    feed_dict={x: X_valid_processed, y: Y_valid, keep_prob: 1})
                
                epoch_list.append(epoch)
                loss_list.append(loss_)
                accuracy_list.append(accuracy_)

                duration = time.time() - start_time

                if epoch % display_step == 0:
                    examples_per_sec = batch_size / duration
                    sec_per_epoch = float(duration)

                    print("Epoch:", '%02d' % (epoch+1), 
                        "cost =", "{:.9f}".format(loss_),
                        "accuracy =", accuracy_,
                        "; %.1f samples/sec; %.3f sec/epoch" % (examples_per_sec, sec_per_epoch))

        print("Optimization Finished!")

        if data_set=='cifar10':
            minibatch_x_test, minibatch_y_test = sess.run([X_test_processed, Y_test])

            if isNB:
                accuracy_ = sess.run(eval_op, feed_dict={x: minibatch_x_test, y: minibatch_y_test, 
                                            keep_prob: 1,
                                            is_train: False})
            else:
                accuracy_ = sess.run(eval_op, feed_dict={x: minibatch_x_test, y: minibatch_y_test, keep_prob: 1})

            print("Test Accuracy:", accuracy_)

        else:

            if isNB:
                print("Test Accuracy:", sess.run(eval_op, feed_dict={x: X_test_processed, y: Y_test, 
                                                            keep_prob: 1,
                                                            is_train: False}))
            else:
                print("Test Accuracy:", sess.run(eval_op, feed_dict={x: X_test_processed, y: Y_test, keep_prob: 1}))               

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
    