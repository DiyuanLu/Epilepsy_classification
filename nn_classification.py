### use basic network to do classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import ipdb

data_dir = "data/sub_train/sub_16"
data_dir_test = "data/sub_test"
seq_len = 640
batch_size = 16
n_outputs1 = 512
n_outputs2 = 128
n_classes = 2
total_batches =  200
pca = PCA(n_components=2)

def get_test_data():
    test_data = np.empty([0, seq_len])
    test_labels = np.empty([0])
    for filen in files_test:
        data = np.average(func.read_data(filen[0]), axis=0)
        test_data = np.vstack((test_data, data))                
        test_labels = np.append(test_labels, filen[1])
    test_labels = np.eye((n_classes))[test_labels.astype(int)]
    return test_data, test_labels
    
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

x = tf.placeholder("float32", [None, seq_len])  #20s recording
y = tf.placeholder("float32")

### construct the network     
def train(x):
    #### Get data
    files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    files_test = func.find_files(data_dir_test, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
    ## create the iterator
    iter = dataset.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename

    outputs = network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct, "float32"))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iter.initializer)
        loss = np.array([])
        accuracy = np.array([])
        test_acc = np.array([])
        test_data, test_labels = get_test_data()

        for batch in range(total_batches):
            filename =  sess.run(ele)   # name, '1'/'0'
            batch_data = np.empty([0, seq_len])
            batch_labels = np.empty([0])
            for ind in range(len(filename)):
                data = np.average(func.read_data(filename[ind][0]), axis=0)
                batch_data = np.vstack((batch_data, data))                
                batch_labels = np.append(batch_labels, filename[ind][1])
                
            batch_labels =  np.eye((n_classes))[batch_labels.astype(int)]
            _, c= sess.run([optimizer, cost], feed_dict={x: batch_data, y: batch_labels})
            loss = np.append(loss, c)
            correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
            ac = tf.reduce_mean(tf.cast(correct, "float32"))
            accuracy = np.append(accuracy, ac)
            test_acc = np.append(test_acc, acc.eval({x:test_data, y:test_labels}))
        plt.figure()
        plt.plot(loss, 'c-')
        plt.xlabel("training batch")
        plt.ylabel("loss")
        plt.plot(test_acc, 'm-')
        plt.xlabel("training batch")
        plt.ylabel("test accuracy")
        plt.show()

if __name__ == "__main__":
    train(x)
### define the cost and opti
