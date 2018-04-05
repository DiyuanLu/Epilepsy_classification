### use basic network to do classification
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import ipdb

data_dir = "data/sub_train/sub_16"
data_dir_test = "data/sub_test"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
version = 'ds16_cnn'
logdir = "results/" + version + '/' + datetime + "/model"
resultdir = "results/" + version + '/' + datetime
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(resultdir ):
    os.makedirs(resultdir )
print resultdir
plot_every = 500
save_every = 500
seq_len = 640
batch_size = 16
n_outputs1 = 1000
n_outputs2 = 500
n_outputs3 = 300
n_classes = 2
total_batches =  10000

def get_test_data(data_dir):
    with tf.name_scope("test_data"):
        files_test = func.find_files(data_dir)
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
                                                                            activation_fn=tf.nn.relu,
                                                                            name='fully1')
    layer2_out = tf.contrib.layers.fully_connected(
                                                                            layer1_out,
                                                                            n_outputs2,
                                                                            activation_fn=tf.nn.relu,
                                                                            name='fully2')
    layer3_out = tf.contrib.layers.fully_connected(
                                                                            layer2_out,
                                                                            n_outputs3,
                                                                            activation_fn=tf.nn.relu,
                                                                            name='fully3')
    layer3_out_bn = tf.contrib.layers.batch_norm(
                                                                        layer3_out,
                                                                        center = True,
                                                                        scale = True)
    outputs = tf.contrib.layers.fully_connected(
                                                                        layer3_out_bn,
                                                                        n_classes,
                                                                        activation_fn=tf.nn.sigmoid)
    return outputs


def CNN(x):
    ## Input layer
    inputs = tf.reshape(x, [-1, 1,  seq_len, 1])
    # Convolutional Layer #1
    with tf.name_scope("conv1"):
        conv1 = tf.layers.conv2d(
                                                    inputs = inputs,
                                                    filters = 32,
                                                    kernel_size = [1, 5],
                                                    padding = 'same',
                                                    activation=tf.nn.relu)
        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)
    with tf.name_scope("conv2"):
        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
                                                    inputs = pool1,
                                                    filters = 64,
                                                    kernel_size = [1, 5],
                                                    padding = 'same',
                                                    activation=tf.nn.relu)
        # Pooling Layer #1
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)
    with tf.name_scope("dense"):
        ## Dense
        pool2_flat = tf.reshape(pool2, [-1,  pool1.shape[1]*pool1.shape[2]*pool1.shape[3]])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4)

        ### Logits layer
        logits = tf.layers.dense(inputs=dropout, units=n_classes)

        return logits

    

x = tf.placeholder("float32", [None, seq_len])  #20s recording
y = tf.placeholder("float32")

### construct the network     
def train(x):
    with tf.name_scope("Data"):
        #### Get data
        files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_test = func.find_files(data_dir_test, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
        dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        ## create the iterator
        iter = dataset.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename

    #### feed the inputs to the network
    #outputs = network(x)
    outputs = CNN(x)
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    with tf.name_scope("performance"):
        correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    trials = 2
    with tf.Session() as sess:
        loss_trial_train = np.zeros([total_batches, trials])    # tracking loss
        acc_trial_train = np.zeros([total_batches, trials])
        acc_trial_test = np.zeros([total_batches, trials])
        test_data, test_labels = get_test_data(data_dir_test)
        outlier = [] 
        for trial in range(trials):
            np.random.seed(1998745)
            sess.run(iter.initializer)   # every trial restart training
            sess.run(tf.global_variables_initializer())
            acc_total_train = np.array([])
            acc_total_test = np.array([])
            loss_total_train = np.array([])
            # track the outlier files
            for batch in range(total_batches):
                filename =  sess.run(ele)   # name, '1'/'0'
                batch_data = np.empty([0, seq_len])
                batch_labels = np.empty([0])
                for ind in range(len(filename)):
                    data = np.average(func.read_data(filename[ind][0]), axis=0)
                    batch_data = np.vstack((batch_data, data))                
                    batch_labels = np.append(batch_labels, filename[ind][1])
                    
                batch_labels =  np.eye((n_classes))[batch_labels.astype(int)]   # get one-hot lable 
                _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels})
                ### record loss and accuracy
                if acc < 4:
                    outlier.append(filename)
                acc_total_train = np.append(acc_total_train, acc)
                acc_total_test = np.append(acc_total_test, accuracy.eval({x:test_data, y:test_labels}))
                loss_total_train = np.append(loss_total_train, c)
                writer.add_summary(summary, batch)
                if batch % save_every == 0:
                    saver.save(sess, logdir + '/batch' + str(batch))
                    
                if batch % plot_every== 0 :
                    func.plotdata(loss_total_train, color='c', ylabel="loss", save_name=resultdir + "/loss_batch_{}".format(batch))
                    func.plotdata(acc_total_test, color='m', ylabel="accuracy", save_name=resultdir + "/test_acc_batch_{}".format(batch))
            acc_trial_train[:, trial] = acc_total_train
            loss_trial_train[:, trial] = loss_total_train
            acc_trial_test[:, trial] = acc_total_test
        # save outliers files name
        np.savetxt(resultdir + "/outlier_files" + ".csv", outliers, delimiter=',')
        func.plot_learning_curve(acc_trial_train, acc_trial_test, save_name=resultdir + "/learning_curve")
        func.plot_smooth_shadow_curve(loss_trial_train, save_name=resultdir + "/loss_in_training")
        
            
if __name__ == "__main__":
    train(x)
### define the cost and opti
