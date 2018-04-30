 ### use basic network to do classification
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
#from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import modules as mod
import ipdb
import time
from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA

mnist = input_data.read_data_sets("data/MNIST_data/",  one_hot=True)

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 500
save_every = 500
height, width = 28, 28 #seq_len, 1     # MNIST
batch_size = 100 # old: 16     20has a very good result
num_classes = 10
epochs = 200
total_batches =  epochs * 3000 // batch_size + 1 #5001               #

pattern='ds_8*.csv'
version = 'whole_MNIST_RNN'              #DilatedCNNDeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4])       #### DeepConvLSTMDeepCLSTM
results_dir= "results/2-MNIST_checks/" + version + '/batch{}/' .format(batch_size)+ datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print results_dir


x = tf.placeholder("float32", [None, height*width])  #20s recording width of each recording is 1, there are 2 channels
y = tf.placeholder("float32")

### construct the network
def train(x):

    #### Constructing the network
    #outputs = mod.fc_net(x, hid_dims=[500, 300], num_classes = num_classes)   ##
    #outputs = mod.resi_net(x, hid_dims=[500, 300], num_classes = num_classes)  ## ok very sfast
    #outputs = mod.CNN(x, num_filters=[32, 64], seq_len=height, width=width, num_classes = num_classes)    ## ok
    #outputs = mod.DeepConvLSTM(x, num_filters=[32, 64], filter_size=5, num_lstm=128, seq_len=height, width=width, num_classes = num_classes)  ## ok
    #outputs = mod.RNN(x, num_lstm=64, seq_len=height, width=width, num_classes = num_classes)   ##ok
    outputs = mod.Dilated_CNN(x, num_filters=8, dilation_rate=[2, 8, 16], kernel_size = [3, 3], pool_size=[2, 2], pool_strides=[2, 2], seq_len=height, width=width, num_classes = num_classes) ##ok
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        

        test_acc = tf.Variable(0.0)
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        test_acc_sum = tf.summary.scalar('test_accuracy', test_acc)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.Session() as sess:
         #profiler = tf.profiler.Profiler(sess.graph)
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        np.random.seed(1998745)
        sess.run(tf.global_variables_initializer())
        acc_total_train = np.array([])
        acc_total_test = np.array([])
        loss_total_train = np.array([])
        for batch in range(total_batches):###
            save_name = results_dir + '/' + "_step{}_".format( batch)
            ########## MNIST
            batch_data, batch_labels = mnist.train.next_batch(batch_size)

            _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels}, options=options, run_metadata=run_metadata)
           
            if batch % 10 == 0:
                print "batch",batch, 'loss', c, 'accuracy', acc
            writer.add_summary(summary, batch)
            ####### # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(save_name + 'timeline_{}.json'.format(batch), 'w') as f:
                f.write(chrome_trace)
                
            if batch % 1 == 0:
                # track training
                acc_total_train = np.append(acc_total_train, acc)
                loss_total_train = np.append(loss_total_train, c)
               
                test_data, test_labels = mnist.test.next_batch(batch_size)
                test_temp = sess.run(accuracy, {x: test_data, y: test_labels})   # test_acc_sum, sensitivity_sum, specificity_sum, 
                acc_total_test = np.append(acc_total_test, test_temp)
                ########################################################
            if batch % save_every == 0:
                saver.save(sess, logdir + '/batch' + str(batch))

            if batch % plot_every == 0 and batch >= plot_every:   #

                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], xlabel= 'training batches / {}'.format(batch_size), ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_batch_{}".format(batch))
                
                func.plot_smooth_shadow_curve([loss_total_train], colors=['c'], xlabel= 'training batches / {}'.format(batch_size), ylabel="loss", title='Loss in training',labels=['training loss'], save_name=results_dir+ "/training_loss_batch_{}".format(batch))

                func.save_data((acc_total_train, loss_total_train, acc_total_test), header='accuracy_train,loss_train,accuracy_test', save_name=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO
        #np.savetxt('outliers.csv', outliers, fmt='%s', newline= ', ', delimiter=',')



if __name__ == "__main__":
    train(x)
### define the cost and opti
