import tensorflow as tf
import numpy as np
import csv
import glob
from PIL import Image



train_size = 0.8
X = np.genfromtxt('X_notes.csv', delimiter=',')
Y = np.genfromtxt('Y_notes.csv', delimiter=',')

m = X.shape[0]
size = X.shape[1]
rand_index = list(np.random.permutation(m))
m_train = int(m*train_size)
X_train = X[rand_index[:m_train],:]
Y_train = np.reshape(Y[rand_index[:m_train]],(m_train,1))
Y_train = np.append(Y_train, 1-Y_train, axis=1)
X_test = X[rand_index[m_train:],:]
Y_test = np.reshape(Y[rand_index[m_train:]],(m-m_train,1))
Y_test = np.append(Y_test, 1-Y_test, axis=1)


x = tf.placeholder(tf.float32, [None, size])
W = tf.Variable(tf.zeros([size, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
sess.run(train_step, feed_dict={x: X_train, y_: Y_train})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print '\n\nFirst Model:\n'
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}))

prediction=tf.argmax(y,1)
print prediction.eval(session = sess, feed_dict={x: X_test})


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
x = tf.placeholder(tf.float32, [None, size])
y_ = tf.placeholder(tf.float32, [None, 2])
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

W_fc2 = weight_variable([20, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
train_step.run(session = sess, feed_dict={x: X_train, y_: Y_train, keep_prob: 0.5})
print '\n\nSecond Model \n'
print("test accuracy %g"%accuracy.eval(session = sess, feed_dict={x: X_test, y_: Y_test, keep_prob: 1.0}))


prediction=tf.argmax(y_conv,1)
print prediction.eval(session = sess, feed_dict={x: X_test, keep_prob: 1.0})


#probabilities=y_conv
#print "probabilities", probabilities.eval(feed_dict={x: X_test, keep_prob:1.0}, session=sess)