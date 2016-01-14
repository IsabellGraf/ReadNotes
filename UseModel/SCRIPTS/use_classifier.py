import tensorflow as tf
import numpy as np
import scipy.misc
import csv
import os
import sys
import glob
import random
from PIL import Image


def get_Note_files_list(this_folder):
    files = sorted(glob.glob(this_folder + '/*.jpg'))
    return files
    

def make_X(files, size_X):
    X = np.zeros(shape=(0, size_X))
    for file_name in files:
        content = np.asarray(Image.open(file_name))
        next_row = np.reshape(content,(1,content.shape[0]*content.shape[1]))/255
        X = np.append(X,next_row,axis=0)
    return X


def get_X_size(file_name):
    content = np.asarray(Image.open(file_name))
    size_X = content.shape[0]*content.shape[1]
    return size_X


def classifier(FILES_FOLDER):
    files = get_Note_files_list(FILES_FOLDER)
    size_X = get_X_size(files[0])
    size_Y = 2

    def weight_variable(shape, this_name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=this_name)

    def bias_variable(shape, this_name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=this_name)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
    b_conv1 = bias_variable([32], 'b_conv1')
    x = tf.placeholder(tf.float32, [None, size_X])
    x_image = tf.reshape(x, [-1,80,32,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
    b_conv2 = bias_variable([64], 'b_conv2')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([20 * 8 * 64, 1024], 'W_fc1')
    b_fc1 = bias_variable([1024], 'b_fc1')

    h_pool2_flat = tf.reshape(h_pool2, [-1, 20 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, size_Y], 'W_fc2')
    b_fc2 = bias_variable([size_Y], 'b_fc2')

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    saver.restore(sess, "../MODEL/model_layers.ckpt")
    print("Model restored.")

    X = make_X(files, size_X)
    prediction=tf.argmax(y_conv,1)
    result = prediction.eval(session = sess, feed_dict={x: X, keep_prob: 1.0})

    return result


def write_result(OUTPUT_FOLDER, result):
    Rf = open(OUTPUT_FOLDER + '/notes_as_numbers.txt','w')
    for yy in result:
        to_save = str(yy) + '\n'
        Rf.write(to_save)
    Rf.close()



if __name__ == '__main__':
    FILES_FOLDER = sys.argv[2]
    OUTPUT_FOLDER = sys.argv[1]
    result = classifier(FILES_FOLDER)
    write_result(OUTPUT_FOLDER, result)

