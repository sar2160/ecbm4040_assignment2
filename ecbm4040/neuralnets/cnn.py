#!/usr/bin/env/ python
# ECBM E4040 Fall 2017 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time

####################################
# TODO: Build your own LeNet model #
####################################
import sys
sys.path.append('.')
from ecbm4040.neuralnets.cnn_sample import * # need funcs
from ecbm4040.image_generator import ImageGenerator



def my_LeNet(input_x, input_y,
          img_len=32, channel_num=3, output_size=10,
          conv_featmap=[6, 16], fc_units=[84, 84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235):
    """
    My version of LeNet, changes include:
    * option to use Adam optimizer
    * added a 3rd fully-connected layer, didn't improve performance so removed it.

    """

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    conv_layer_1 = conv_layer(input_x=pooling_layer_0,
                              in_channel=channel_num,
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")

    # flatten
    pool_shape = pooling_layer_1.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_1.output(), shape=[-1, img_vector_length])

    # fc layer, adding a third
    
    
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[1],
                          out_size=units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1)



    # saving the parameters for l2_norm loss
    conv_w = [conv_layer_0.weight, conv_layer_1.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.norm(w, axis=[-2, -1]) for w in conv_w])

        label = tf.one_hot(input_y, 10)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_1.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('LeNet_loss', loss)

    return fc_layer_1.output(), loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 10)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


## adding option for ADAM
def train_step(loss, learning_rate=1e-3, adam=False):
    if adam:        
        with tf.name_scope('train_step'):
            step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    else:
        with tf.name_scope('train_step'):
            step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return step



def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num


    
    
####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################




# my training function
def my_training(X_train, y_train, X_val, y_val, 
             conv_featmap=[6, 6],
             fc_units=[84, 84],
             conv_kernel_size=[5, 5],
             pooling_size=[2, 2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             use_adam=True):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = LeNet(xs, ys,
                         img_len=32,
                         channel_num=3,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))
    
   
        
    
    step = train_step(loss, learning_rate = learning_rate, adam = use_adam) # decide to use ADAM
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    
    # saving parameter settings
    cache = [cur_model_name,
             conv_featmap,
             fc_units,
             conv_kernel_size,
             pooling_size,
             l2_norm,
             seed,
             learning_rate,
             epoch,
             batch_size,
             verbose,
             pre_trained_model,
             use_adam]
    
    # return results and parameters
    return best_acc, cache
    
    
    
    
##########################################
#            End of your code            #
##########################################

def reshapeImg(X,idx):
    img_flat = X[idx,:].flatten()
    img_R = img_flat[0:1024].reshape((32, 32)) / 255
    img_G = img_flat[1024:2048].reshape((32, 32)) / 255
    img_B = img_flat[2048:3072].reshape((32, 32)) /255
    img = np.dstack((img_R, img_G, img_B))
    return img

def reshapeArray(X):
    container = np.ndarray((X.shape[0],32,32,3))
    for n in range(X.shape[0]):
        container[n] = reshapeImg(X,n)

    return container



def my_training_task4(Train, X_val, y_val, 
             conv_featmap=[6, 6],
             fc_units=[84, 84],
             conv_kernel_size=[5, 5],
             pooling_size=[2, 2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None,
             use_adam=True):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)

    output, loss = LeNet(xs, ys,
                         img_len=32,
                         channel_num=3,
                         output_size=10,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(Train.x.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))
    
   
        
    
    step = train_step(loss, learning_rate = learning_rate, adam = use_adam) # decide to use ADAM
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))
    

    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        
        

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

                                                                                        
        train_generator = Train.next_batch_gen(batch_size, shuffle = False)
            
            
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            
            for itr in range(iters):
                iter_total += 1
                
                (training_batch_x, training_batch_y) = next(train_generator)

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            Train.x.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
    
    # saving parameter settings
    cache = [cur_model_name,
             conv_featmap,
             fc_units,
             conv_kernel_size,
             pooling_size,
             l2_norm,
             seed,
             learning_rate,
             epoch,
             batch_size,
             verbose,
             pre_trained_model,
             use_adam]
    
    # return results and parameters
    return best_acc, cache
