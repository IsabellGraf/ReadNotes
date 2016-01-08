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
    Y[range(m),list(Y_index)] = 1

    rand_index1 = list(np.random.permutation(m))
    X = X[rand_index1,:]
    Y = Y[rand_index1,:]
    rand_index2 = list(np.random.permutation(m))
    X = X[rand_index2,:]
    Y = Y[rand_index2,:]

    m_train = int(m*train_size)
    X_train = X[:m_train,:]
    X_test = X[m_train:,:]    
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

    x = tf.placeholder(tf.float32, [None, size_X])
    W = tf.Variable(tf.zeros([size_X, size_Y]))
    b = tf.Variable(tf.zeros([size_Y]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, size_Y])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y + 1e-9))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for X_train, Y_train in zip(X_list, Y_list):
        sess.run(train_step, feed_dict={x: X_train, y_: Y_train})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print '\n\nFirst Model:\n'
    print("test accuracy %g" % sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}))

    prediction=tf.argmax(y,1)
    result = prediction.eval(session = sess, feed_dict={x: X_test})
    print "prediction: ", result
    to_save = Y_test, result
    np.savetxt("result.csv", to_save, delimiter=",")

    #probabilities=y
    #print "probabilities", probabilities.eval(session = sess, feed_dict={x: X_test})

if __name__ == '__main__':
    IN_FILE_X = sys.argv[1]
    IN_FILE_Y = sys.argv[2]
    X_train,Y_train,X_test,Y_test = get_X_Y(IN_FILE_X, IN_FILE_Y)
    X_list, Y_list = divide_XY_in_batches(X_train,Y_train)
    classifier(X_list,Y_list,X_test,Y_test)
    