import tensorflow as tf
import numpy as np
import glob
import scipy.misc
import sys

def only0and1(batch):
    X9 = batch[0]
    Y9 = batch[1]
    x_size = X9.shape[1]
    X1 = np.zeros(shape=(0,x_size,))
    Y1 = np.zeros(shape=(0,2))

    for x,y in zip(X9,Y9):
        if y[0]==1:
            Y1 = np.concatenate((Y1,np.array([[1, 0]])), axis=0)
            X1 = np.concatenate((X1,np.reshape(x,(1,x_size))), axis=0)
        if y[1]==1:
            Y1 = np.concatenate((Y1,np.array([[0, 1]])), axis=0)
            X1 = np.concatenate((X1,np.reshape(x,(1,x_size))), axis=0)
    return X1, Y1





from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X, y = mnist.train.next_batch(100)

#####  erstes Model #######

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
    batch = mnist.train.next_batch(100)
    batch_xs, batch_ys = only0and1(batch)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


test_data = mnist.test.images, mnist.test.labels
x_test,y_test = only0and1(test_data)
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

result = sess.run(y, feed_dict={x: x_test})
#for image,yy in zip(x_test,y_test):
#    zahl = np.reshape(image,(28,28))
#    num_files = len(glob.glob('./*.jpg'))
#    scipy.misc.imsave('note%04i_%01i.jpg' % (num_files,round(yy[1])), zahl)
#np.savetxt("aimage.csv", x_test[19], delimiter=",")
sys.exit()

#####  zweites Model #######

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

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
for i in range(200):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        print type(batch[0])
        print batch[0].shape
        print type(batch[1])
        print batch[1].shape
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session = sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session = sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


prediction=tf.argmax(y_conv,1)
print prediction.eval(session = sess, feed_dict={x: mnist.test.images, keep_prob: 1.0})


probabilities=y_conv
print "probabilities", probabilities.eval(feed_dict={x: mnist.test.images, keep_prob: 1.0}, session=sess)