import tensorflow as tf
import numpy as np
import csv
import sys
from PIL import Image


def get_X_Y(IN_FILE_X, IN_FILE_Y):
    train_size = 0.9
    X = np.genfromtxt(IN_FILE_X, delimiter=',')
    Y_index = np.genfromtxt(IN_FILE_Y, delimiter=',')

    m = X.shape[0]
    size_Y = max(Y_index)+1
    size_X = X.shape[1]

    Y = np.zeros((m, size_Y))

    rand_index = list(np.random.permutation(m))
    m_train = int(m*train_size)
    X_train = X[rand_index[:m_train],:]
    X_test = X[rand_index[m_train:],:]
    
    Y[range(m),list(Y_index[rand_index])] = 1
    
    Y_train = Y[:m_train,:]
    Y_test = Y[m_train:,:]
    return X_train,Y_train,X_test,Y_test

def divide_XY_in_batches(X_train,Y_train):
    batch_size = 100
    m = X_train.shape[0]
    X_list = []
    Y_list = []
    how_many = int(m/batch_size)
    begin = 0
    for i in range(how_many):
        end = (i+1)*batch_size
        X_list.append(X_train[begin:end,:])
        Y_list.append(Y_train[begin:end,:])
        begin = end
    if m - begin > 0:
        X_list.append(X_train[begin:m,:])
        Y_list.append(Y_train[begin:m,:])
    return X_list, Y_list



def classifier(X_list,Y_list,X_test,Y_test):
    size_X = X_test.shape[1]
    size_Y = Y_test.shape[1]

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 20])
    b_conv1 = bias_variable([20])
    x = tf.placeholder(tf.float32, [None, size_X])
    y_ = tf.placeholder(tf.float32, [None, size_Y])
    x_image = tf.reshape(x, [-1,32,80,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 20, 20])
    b_conv2 = bias_variable([20])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([8 * 20 * 20, 20])
    b_fc1 = bias_variable([20])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 20 * 20])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([20, size_Y])
    b_fc2 = bias_variable([size_Y])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print '\n\nSecond Model \n'

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    for X_train, Y_train in zip(X_list, Y_list):
        train_step.run(session = sess, feed_dict={x: X_train, y_: Y_train, keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(session = sess, feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}))
    prediction=tf.argmax(y_conv,1)
    print prediction.eval(session = sess, feed_dict={x: X_test, keep_prob: 1.0})


    #probabilities=y_conv
    #print "probabilities", probabilities.eval(feed_dict={x: X_test, keep_prob:1.0}, session=sess)


if __name__ == '__main__':
    IN_FILE_X = sys.argv[1]
    IN_FILE_Y = sys.argv[2]
    X_train,Y_train,X_test,Y_test = get_X_Y(IN_FILE_X, IN_FILE_Y)
    X_list, Y_list = divide_XY_in_batches(X_train,Y_train)
    classifier(X_list,Y_list,X_test,Y_test)