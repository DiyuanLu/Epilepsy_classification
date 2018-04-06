### use basic network to do classification
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import modules as mod
import ipdb

data_dir = "data/sub_train/sub_8"
data_dir_test = "data/sub_test/sub_8"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
version = 'ds8_2cnn'
logdir = "results/" + version + '/' + datetime + "/model"
resultdir = "results/" + version + '/' + datetime
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(resultdir ):
    os.makedirs(resultdir )
print resultdir
plot_every = 500
save_every = 500
seq_len = 1280
batch_size = 16
n_classes = 2
total_batches =  50001

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


def RNN(x, rnn_size=500):
    rnn_size = rnn_size
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)   # new version
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    
    return output

    
x = tf.placeholder("float32", [None, seq_len])  #20s recording
y = tf.placeholder("float32")

### construct the network     
def train(x):
    with tf.name_scope("Data"):
        #### Get data
        files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_test = func.find_files(data_dir_test, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
        file_tensor_test = tf.convert_to_tensor(files_test, dtype=tf.string)## convert to tensor
        dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_test = tf.data.Dataset.from_tensor_slices(file_tensor_test).repeat().batch(batch_size).shuffle(buffer_size=10000)
        ## create the iterator
        iter = dataset.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        ele_test = iter_test.get_next()   #you get the filename

    #### feed the inputs to the network
    #outputs = mod.network(x)
    outputs = mod.CNN(x, seq_len=seq_len)
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y))
    with tf.name_scope("performance"):
        correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
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
        outliers = [] 
        for trial in range(trials):
            #np.random.seed(1998745)
            sess.run(iter.initializer)   # every trial restart training
            sess.run(iter_test.initializer)   # every trial restart training
            sess.run(tf.global_variables_initializer())
            acc_total_train = np.array([])
            acc_total_test = np.array([])
            loss_total_train = np.array([])
            # track the outlier files
            for batch in range(total_batches):
                filename =  sess.run(ele)   # name, '1'/'0'
                filename_test =  sess.run(ele_test)   # name, '1'/'0'
                batch_data = np.empty([0, seq_len])
                batch_labels = np.empty([0])
                for ind in range(len(filename)):
                    data = np.average(func.read_data(filename[ind][0]), axis=0)
                    batch_data = np.vstack((batch_data, data))                
                    batch_labels = np.append(batch_labels, filename[ind][1])
                batch_labels =  np.eye((n_classes))[batch_labels.astype(int)]   # get one-hot lable
                
                _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels})
                ### record loss and accuracy
                if acc < 0.35:
                    outliers.append(filename)
                acc_total_train = np.append(acc_total_train, acc)
                ### test
                test_data = np.empty([0, seq_len])
                test_labels = np.empty([0])
                for ind in range(len(filename_test)):
                    data = np.average(func.read_data(filename_test[ind][0]), axis=0)
                    test_data = np.vstack((test_data, data))                
                    test_labels = np.append(test_labels, filename_test[ind][1])
                test_labels =  np.eye((n_classes))[test_labels.astype(int)]   # get one-hot lable
                
                acc_total_test = np.append(acc_total_test, accuracy.eval({x:test_data, y:test_labels}))
                loss_total_train = np.append(loss_total_train, c)
                writer.add_summary(summary, batch)
                if batch % save_every == 0:
                    saver.save(sess, logdir + '/batch' + str(batch))
                    
                if batch % plot_every == 0 :
                    func.plotdata(loss_total_train, color='c', ylabel="loss", save_name=resultdir + "/loss_batch_{}".format(batch))
                    func.plotdata(acc_total_test, color='m', ylabel="accuracy", save_name=resultdir + "/test_acc_batch_{}".format(batch))
            acc_trial_train[:, trial] = acc_total_train
            loss_trial_train[:, trial] = loss_total_train
            acc_trial_test[:, trial] = acc_total_test
          
        # save outliers files name
        #np.savetxt(resultdir + "/outlier_files" + ".txt", outliers, delimiter=',')
        ipdb.set_trace()
        func.plot_learning_curve(acc_trial_train, acc_trial_test, save_name=resultdir + "/learning_curve")
        func.plot_smooth_shadow_curve(loss_trial_train, save_name=resultdir + "/loss_in_training")
        
            
if __name__ == "__main__":
    train(x)
### define the cost and opti
