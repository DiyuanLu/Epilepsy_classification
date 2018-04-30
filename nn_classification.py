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

#data_dir = "data/train_data"
data_dir = "data/test_files/test_tf"
#data_dir_test = "data/test_data"
data_dir_test = "data/test_files/test_tf"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 200
save_every = 100
seq_len = 1280   ##10240  #
height = seq_len
ifaverage = False
if  ifaverage:
    width= 1
else:
    width = 2

batch_size = 20 # old: 16     20has a very good result
num_classes = 2
epochs = 200
total_batches =  epochs * 3000 // batch_size + 1 #5001               #
num_classes = 2
pattern='ds_8*.csv'
version = 'whole_{}_DilatedCNN'.format(pattern[0:4])                    #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4])       #### DeepConvLSTMDeepCLSTMDilatedCNN
results_dir= "results/" + version + '/cpu-batch{}/' .format(batch_size)+ datetime
logdir = results_dir+ "/model"

print results_dir


x = tf.placeholder("float32", [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
y = tf.placeholder("float32")

### construct the network
def train(x):
    #ele, ele_test, iter, iter_test = func.load_train_test_data(data_dir, data_dir_test, batch_size=2, pattern=pattern, withlabel=True)
    #data_train, labels_train, data_test, labels_test = func.load_train_test_data_queue(data_dir, data_dir_test,  batch_size=20, pattern='Data*.csv', withlabel=True)
    
    with tf.name_scope("Data"):
        #### Get data
        files_wlabel_train = func.find_files(data_dir, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_wlabel_test = func.find_files(data_dir_test, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        #files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
        #files_test, labels_test = np.array(files_wlabel_test)[:, 0], np.array(files_wlabel_test)[:, 1].astype(np.int)   ## seperate the name and label
        dataset_train = tf.data.Dataset.from_tensor_slices(files_wlabel_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_test = tf.data.Dataset.from_tensor_slices(files_wlabel_test).repeat().batch(batch_size).shuffle(buffer_size=10000)
        ## create TensorFlow Dataset objects
        #dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        #dataset_test = tf.data.Dataset.from_tensor_slices((files_test, labels_test)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        #### map self-defined functions to the dataset
        
        #dataset_train = dataset_train.map(func.input_parser)
        #dataset_test = dataset_test.map(func.input_parser)
       
        iter = dataset_train.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        ele_test = iter_test.get_next()   #you get the filename

    ################ Constructing the network ###########################
    #outputs = mod.fc_net(x, hid_dims=[500, 300], num_classes = num_classes)   ##
    #outputs = mod.resi_net(x, hid_dims=[500, 300], num_classes = num_classes)  ## ok very sfast
    #outputs = mod.CNN(x, num_filters=[32, 64], seq_len=height, width=width, num_classes = num_classes)    ## ok
    #outputs = mod.DeepConvLSTM(x, num_filters=[32, 64], filter_size=5, num_lstm=128, seq_len=height, width=width, num_classes = num_classes)  ## ok
    #outputs = mod.RNN(x, num_lstm=32, seq_len=height, width=width, num_classes = num_classes)   ##ok
    outputs = mod.Dilated_CNN(x, num_filters=16, seq_len=seq_len, width=width, num_classes = num_classes)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(y, 1), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        #accuracy_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(outputs, 1), tf.argmax(y, 1), num_classes, name='accuracy_per_class')
        ######sensitiity = TP / TP + FN, specificity = TN / TN + FP, apc =[TN, FN], [TP, FP]]
        #sensitivity = accuracy_per_class[1][1, 0] / (accuracy_per_class[1][1, 0] + accuracy_per_class[1][0, 1] )   ## false_positive / batch_size
        #specificity = accuracy_per_class[1][0, 0] / (accuracy_per_class[1][0, 0] + accuracy_per_class[1][1, 1])   ## true_positive / batch_size

        test_acc = tf.Variable(0.0)
        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        test_acc_sum = tf.summary.scalar('test_accuracy', test_acc)

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)###, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        #### Profiling
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        outliers = []
        np.random.seed(1998745)
        sess.run(iter.initializer)   # every trial restart training
        sess.run(iter_test.initializer)    # every trial restart training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        acc_total_train = np.array([])
        acc_total_test = np.array([])
        loss_total_train = np.array([])
        # track the outlier files
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        for batch in range(total_batches):#####3
            save_name = results_dir + '/' + "_step{}_".format( batch)
            
            #data_train, labels_train, data_test, labels_test = sess.run([data_train, labels_train, data_test, labels_test])
           
            filename =  sess.run(ele)   # (name, 1/0)
            filename_test =  sess.run(ele_test)   # (name, 1/0)
            #batch_data, batch_labels = input_parser(file_path, label, num_classes=2)
            batch_data = []
            batch_labels = np.empty([0])
            for ind in range(len(filename)):
                data = func.read_data(filename[ind][0], ifaverage=ifaverage)
                batch_data.append(data)
                batch_labels = np.append(batch_labels, filename[ind][1])
            batch_labels =  np.eye((num_classes))[batch_labels.astype(int)]   # get one-hot lable
            #batch_data = np.expand_dims(batch_data, axis=2)  # from shape [None, seq_len, 2] to [None, 1, seq_len, 2]
            ##ipdb.set_trace()
            
            _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: batch_data, y: batch_labels}, options=options, run_metadata=run_metadata)
            # We collect profiling infos for each step.
            writer.add_summary(summary, batch)
            
            ####### # Create the Timeline object, and write it to a json file
            #fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #with open(save_name + 'timeline_{}.json'.format(batch), 'w') as f:
                #f.write(chrome_trace)


            ### record loss and accuracy
            #if acc < 0.35:
                #outliers.append(filename)
            #ipdb.set_trace()
            if batch % 10 == 0:
                # track training
                acc_total_train = np.append(acc_total_train, acc)
                #sen_total_train = np.append(sen_total_train, sensi)   # sensitivity
                #spe_total_train = np.append(spe_total_train, speci)
                loss_total_train = np.append(loss_total_train, c)
                ############################################################ test
                test_data = []
                test_labels = np.empty([0])
                for ind in range(len(filename_test)):
                    data = func.read_data(filename_test[ind][0], ifaverage=ifaverage)
                    test_data.append(data)
                    test_labels = np.append(test_labels, filename_test[ind][1])
                test_labels =  np.eye((num_classes))[test_labels.astype(int)]   # get one-hot lable

                test_temp = sess.run(accuracy, {x: test_data, y: test_labels})   # test_acc_sum, sensitivity_sum, specificity_sum,
                acc_total_test = np.append(acc_total_test, test_temp)
                ########################################################
            if batch % 10 == 0:
                print "batch:",batch, 'loss:', c, 'train-accuracy:', acc, 'test-accuracy:', test_temp
            if batch % save_every == 0:
                saver.save(sess, logdir + '/batch' + str(batch))

            if batch % plot_every == 0 and batch >= plot_every:   #

                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], xlabel= 'training batches / {}'.format(batch_size), ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_batch_{}".format(batch))

                func.plot_smooth_shadow_curve([loss_total_train], colors=['c'], xlabel= 'training batches / {}'.format(batch_size), ylabel="loss", title='Loss in training',labels=['training loss'], save_name=results_dir+ "/training_loss_batch_{}".format(batch))

                func.save_data((acc_total_train, loss_total_train, acc_total_test), header='accuracy_train,loss_train,accuracy_test', save_name=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO
        #np.savetxt('outliers.csv', outliers, fmt='%s', newline= ', ', delimiter=',')
    coord.join(threads)


if __name__ == "__main__":
    train(x)
### define the cost and opti
