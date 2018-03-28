### use basic network to do classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import fnmatch
import ipdb
import csv
import codecs

seq_len = 10240
batch_size = 16
n_outputs1 = 512
n_outputs2 = 128
n_classes = 2
total_batches =  2000


def find_files(directory, pattern='*.csv'):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
#             print filenames
            if 'Data_F' in filename:
                label = '1'
            elif 'Data_N' in filename:
                label = '0'
            files.append((os.path.join(root, filename), label))
                    
    return files

def read_data(filename ):
    reader = csv.reader(codecs.open(filename, 'rb', 'utf-8'))
    x_data, y_data = np.array([]), np.array([])
    for ind, row in enumerate(reader):
        x, y = row
        x_data = np.append(x_data, x)
        y_data = np.append(y_data, y)
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    return x_data, y_data
#### Get data
#def get_next_batch(data, batch_size, Focal=True):
data_dir = "data/train_data"
files = []
### traverse all the files in the dir, and divide into batches, from WaveNet
files_train = find_files(data_dir )
## convert to tensor
file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)
## create dataset from tensor
dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
## create the iterator
iter = dataset.make_initializable_iterator()
ele = iter.get_next()   #you get the filename

    #return ele   # batch_size * (filename, label)


### load the data
x = tf.placeholder("float32", [batch_size, seq_len])  #20s recording
y = tf.placeholder("float32")


### construct the network
def network(x):
    layer1_out = tf.contrib.layers.fully_connected(
                                                                            x,
                                                                            n_outputs1,
                                                                            activation_fn=tf.nn.relu)
    layer2_out = tf.contrib.layers.fully_connected(
                                                                            layer1_out,
                                                                            n_outputs2,
                                                                            activation_fn=tf.nn.relu)
    outputs = tf.contrib.layers.fully_connected(
                                                                            layer2_out,
                                                                            n_classes,
                                                                            activation_fn=tf.nn.sigmoid)
    return outputs
                                                                    
def train(x):
    outputs = network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iter.initializer)
        loss = np.array([])
        accuracy = np.array([])
        for batch in range(total_batches):
            filename =  sess.run(ele)   # name, '1'/'0'
            batch_data = np.empty([0, 10240])
            batch_labels = np.empty([0])
            for ind in range(len(filename)):
                data = np.average(read_data(filename[ind][0]), axis=0)
                batch_data = np.vstack((batch_data, data))
                batch_labels = np.append(batch_labels, filename[ind][1])
            batch_labels =  np.eye((n_classes))[batch_labels.astype(int)]
            _, c, out = sess.run([optimizer, cost, outputs], feed_dict={x: batch_data, y:batch_labels})
            loss = np.append(loss, c)
            print("Batch:", batch+1, "filename", filename, "loss:", c , "outputs", out)

            if batch % 1== 0:
                correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
                acc = tf.reduce_mean(tf.cast(correct, "float32"))
                accuracy = np.append(accuracy, acc)
                
                #print("Accuracy:", accuracy.eval({x:test_x, y:test_y}))
        plt.figure()
        plt.plot(loss, 'c*-')
        plt.show()

if __name__ == "__main__":
    train(x)
### define the cost and opti
