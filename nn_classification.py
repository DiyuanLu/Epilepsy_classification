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
import time



data_dir = "data/train_data"
data_dir_test = "data/test_data"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
version = 'whole_batch20_ds8_ori_cnn3'
pattern='*ds_8.csv'
results_dir= "results/" + version + '/' + datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print results_dir
plot_every = 100
save_every = 100
seq_len = 1280   #10240  ##
batch_size = 20  # old: 16     20has a very good result
n_classes = 2
epochs = 10
total_batches =  epochs * 3000 // batch_size + 1

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
        files_train = func.find_files(data_dir, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_test = func.find_files(data_dir_test, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
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
    #outputs = mod.CNN2(x, seq_len=seq_len)
    outputs = mod.CNN(x, seq_len=seq_len, num_filters=[32, 64])

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        #test_acc = tf.placeholder(tf.float32, name="test_acc")    # track acc in test
        accuracy_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(outputs, 1), tf.argmax(y, 1), n_classes, name='accuracy_per_class')
        # sensitiity = TP / TP + FN, specificity = TN / TN + FP, apc =[TN, FN], [TP, FP]]
        sensitivity = accuracy_per_class[1][1, 0] / (accuracy_per_class[1][1, 0] + accuracy_per_class[1][0, 1] )   ## false_positive / batch_size
        specificity = accuracy_per_class[1][0, 0] / (accuracy_per_class[1][0, 0] + accuracy_per_class[1][1, 1])   ## true_positive / batch_size
        ###### Mask out the padded frames
        #loss_neg = tf.reduce_mean(1 - specificity)
        #loss_posi = tf.reduce_mean(1 - sensitivity)
        #d_loss = tf.reduce_mean(loss_neg) + tf.reduce_mean(loss_neg)  # This optimizes the discriminator.
        
        test_acc = tf.Variable(0.0)
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        sensitivity_sum = tf.summary.scalar('sensitivity', sensitivity)
        specificity_sum = tf.summary.scalar('specificity', specificity)
        test_acc_sum = tf.summary.scalar('test_accuracy', test_acc)

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    trials = 1
    with tf.Session() as sess:
        loss_trial_train = np.zeros([total_batches, trials])    # tracking loss
        acc_trial_train = np.zeros([total_batches, trials])
        acc_trial_test = np.zeros([total_batches, trials])
        #test_data, test_labels = get_test_data(data_dir_test)
        outliers = []
        for trial in range(trials):
            #np.random.seed(1998745)
            sess.run(iter.initializer)   # every trial restart training
            sess.run(iter_test.initializer)   # every trial restart training
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            acc_total_train = np.array([])
            acc_total_test = np.array([])
            loss_total_train = np.array([])
            sen_total_train = np.array([])   # sensitivity
            spe_total_train = np.array([])    # specificity
            # track the outlier files
            for batch in range(total_batches):
                save_name = results_dir + '/' + "_step{}_".format( batch)
                filename =  sess.run(ele)   # name, '1'/'0'
                filename_test =  sess.run(ele_test)   # name, '1'/'0'

                batch_data = np.empty([0, seq_len])
                batch_labels = np.empty([0])
                for ind in range(len(filename)):
                    data = np.average(func.read_data(filename[ind][0]), axis=0)
                    batch_data = np.vstack((batch_data, data))
                    batch_labels = np.append(batch_labels, filename[ind][1])
                batch_labels =  np.eye((n_classes))[batch_labels.astype(int)]   # get one-hot lable

                _, acc, c, sensi, speci, summary = sess.run([optimizer, accuracy, cost, sensitivity, specificity, summaries], feed_dict={x: batch_data, y: batch_labels})
                if batch % 100 == 0:
                    print "trial: ", trial, "batch",batch, "sensitivity", sensi, "specificity", speci
                writer.add_summary(summary, batch)
                ### record loss and accuracy
                if acc < 0.35:
                    outliers.append(filename)
                if batch % 1 == 0:
                    acc_total_train = np.append(acc_total_train, acc)
                    sen_total_train = np.append(sen_total_train, sensi)   # sensitivity
                    spe_total_train = np.append(spe_total_train, speci)
                    loss_total_train = np.append(loss_total_train, c)
                    ############################################################ test
                    test_data = np.empty([0, seq_len])
                    test_labels = np.empty([0])
                    for ind in range(len(filename_test)):
                        data = np.average(func.read_data(filename_test[ind][0]), axis=0)
                        test_data = np.vstack((test_data, data))
                        test_labels = np.append(test_labels, filename_test[ind][1])
                    test_labels =  np.eye((n_classes))[test_labels.astype(int)]   # get one-hot lable
                    test_temp = accuracy.eval({x:test_data, y:test_labels})
                    acc_total_test = np.append(acc_total_test, test_temp)
                    summary = sess.run(summaries, {x:test_data, y:test_labels})   # test_acc_sum, sensitivity_sum, specificity_sum, 
                    summary = sess.run(test_acc_sum, {test_acc: test_temp})    ## add test score to summary
                    writer.add_summary(summary, batch)
                    ########################################################
                if batch % save_every == 0:
                    saver.save(sess, logdir + '/batch' + str(batch))

                if batch % plot_every == 0 and batch > plot_every :   #
                    func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_batch_{}".format(batch))

                    func.plot_smooth_shadow_curve(loss_total_train, colors='c', ylabel="loss", title='Loss in training',labels='loss_train', save_name=results_dir+ "/loss_batch_{}".format(batch))
                    #ipdb.set_trace()
                    func.save_data((acc_total_train, loss_total_train, acc_total_test, sen_total_train, spe_total_train), header='accuracy_train,loss_train,accuracy_test,sen_total_train,spe_total_train', save_dir=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO
            acc_trial_train[:, trial] = acc_total_train
            loss_trial_train[:, trial] = loss_total_train
            acc_trial_test[:, trial] = acc_total_test



if __name__ == "__main__":
    train(x)
### define the cost and opti
