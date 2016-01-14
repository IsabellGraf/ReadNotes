import tensorflow as tf
import numpy as np
import csv
import sys
import os
import glob
import random
from PIL import Image


def get_Note_files_list(this_folder):
    train_size = 0.9
    folders = sorted([x[0] for x in os.walk(this_folder)])[1:]
    files = []
    for folder in folders:
            files = files + glob.glob(folder + '/*.jpg')
    files = random.sample(files, len(files))
    files = random.sample(files, len(files))

    m_train = int(len(files)*train_size)
    return files[:m_train], files[m_train:]


def make_batches(files_list):
    batch_size = 50
    m = len(files_list)
    how_many = int(m/batch_size)
    Files_batches = []
    begin = 0
    for i in range(how_many):
        end = (i+1)*batch_size
        Files_batches.append(files_list[begin:end])
        begin = end
    if m - begin > 0:
        Files_batches.append(files_list[begin:m])
    return Files_batches
    

def make_XY(batch, size_X):
    X = np.zeros(shape=(0, size_X))
    Y = np.zeros(shape=(0, 2))
    for file_name in batch:
        content = np.asarray(Image.open(file_name))
        next_row = np.reshape(content,(1,content.shape[0]*content.shape[1]))/255
        X = np.append(X,next_row,axis=0)
        if '/YES/' in file_name:
            next_row = np.array([0, 1])
            next_row = np.reshape(next_row,(1,2))
            Y = np.append(Y,next_row, axis=0)
        else:
            next_row = np.array([1, 0])
            next_row = np.reshape(next_row,(1,2))
            Y = np.append(Y,next_row, axis=0)
    return X, Y


def get_X_size(file_name):
    content = np.asarray(Image.open(file_name))
    size_X = content.shape[0]*content.shape[1]
    return size_X


def classifier(FILES_FOLDER):
    
    Files_list_train, Files_list_test = get_Note_files_list(FILES_FOLDER)
    Files_batches = make_batches(Files_list_train)
    size_X = get_X_size(Files_list_test[0])
    size_Y = 2

    x = tf.placeholder(tf.float32, [None, size_X])
    W = tf.Variable(tf.zeros([size_X, size_Y]))
    b = tf.Variable(tf.zeros([size_Y]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, size_Y])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y + 1e-9)) # logistic regression as cost function
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    i=0
    for batch in Files_batches:
        X_train, Y_train = make_XY(batch, size_X)
        if i%10 == 0:
            print("Training accuracy %g" % sess.run(accuracy, feed_dict={x: X_train, y_: Y_train}))
        sess.run(train_step, feed_dict={x: X_train, y_: Y_train})
        i=i+1

    print '\n\nFirst Model:\n'
    X_test, Y_test = make_XY(Files_list_test, size_X)
    print("test accuracy %g" % sess.run(accuracy, feed_dict={x: X_test, y_: Y_test}))

    prediction=tf.argmax(y,1)
    result = prediction.eval(session = sess, feed_dict={x: X_test})
    print "prediction: ", result

    Rf = open('result.txt','w')
    for xx,yy in zip(Files_list_test, result):
        to_save = xx + ' ' + str(yy) + '\n'
        Rf.write(to_save)
    Rf.close()
    
    if not os.path.exists('../MODEL/'):
        os.makedirs('../MODEL/')

    save_path = saver.save(sess, "../MODEL/model.ckpt")

    #probabilities=y
    #print "probabilities", probabilities.eval(session = sess, feed_dict={x: X_test})

if __name__ == '__main__':
    FILES_FOLDER = sys.argv[1]
    classifier(FILES_FOLDER)
    