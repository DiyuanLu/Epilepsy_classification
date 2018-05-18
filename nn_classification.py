 ### use basic network to do classification
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
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
from scipy.stats import zscore
data_dir = "data/train_data"
#data_dir_test = "data/test_data"
data_dir_test = "data/npy_data/testF751testN1501_ori_norm_zscore.npz"

#ipdb.set_trace()
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 200
save_every = 100
test_every = 10
smooth_win_len = 20
seq_len = 10240  #1280   ##
height = seq_len
width = 2   ### data width
num_seg = 5   ## number of shorter segments you want to divide the original long sequence with a sliding window 
ifnorm = True
batch_size = 30 # old: 16     20has a very good result
num_classes = 2
epochs = 200
total_batches =  epochs * 3000 // batch_size + 1 #5001               #
num_classes = 2
pattern='Data*.csv'
version = 'whole_{}_Atrous_CNN'.format(pattern[0:4])                    #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4])       #### DeepConvLSTMDeepCLSTMDilatedCNN
results_dir= "results/" + version + '/cpu-batch{}/ori_' .format(batch_size)+ datetime
logdir = results_dir+ "/model"
ifslide = True
if ifslide:   ### use a 5s window slide over the 20s recording and do classification on segments, and then do a average vote
    height = height / num_seg
    x = tf.placeholder("float32", [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
else:
    x = tf.placeholder("float32", [None, height, width])
y = tf.placeholder("float32")

def postprocess(prediction, num_seg=4, Threshold=0.5):
    '''post process the prediction label. average among all the segments of one sequence. Since it is binary classification, Threshold as 0.5 and the average could work
    paramd:
        prediction: the int label prdiction
        num_seg: the number of segments in one sequence
        threshold: used to binarize two-class classification
    return:
        pred: the binarized averaged prediction for whoel sequence'''
    post = tf.reduce_mean(tf.cast(tf.reshape(prediction, [-1, num_seg]), tf.float32), 1)   
    pred = tf.to_int64(post > Threshold)
    return pred   ### every 20s recording segmented into num_seg shorter recordings

### construct the network
def train(x):
    
    with tf.name_scope("Data"):
        ### Get data
        files_wlabel_train = func.find_files(data_dir, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
        dataset_train = tf.data.Dataset.from_tensor_slices(files_wlabel_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        iter = dataset_train.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename

    ################# Constructing the network ###########################
    ##outputs = mod.fc_net(x, hid_dims=[500, 300], num_classes = num_classes)   ##
    ##outputs = mod.resi_net(x, hid_dims=[500, 300], num_classes = num_classes)  ## ok very sfast
    outputs = mod.CNN(x, num_filters=[16, 32, 64], num_block=4, seq_len=height, width=width, num_classes = num_classes)    ## ok
    ##outputs = mod.DeepConvLSTM(x, num_filters=[32, 64], filter_size=5, num_lstm=128, seq_len=height, width=width, num_classes = num_classes)  ## ok
    ##outputs = mod.RNN(x, num_lstm=32, seq_len=height, width=width, num_classes = num_classes)   ##ok
    #outputs = mod.Dilated_CNN(x, num_filters=16, seq_len=seq_len, width=width, num_classes = num_classes)
    #outputs = mod.Atrous_CNN(x, num_filters_cnn=[4, 8, 16, 32], dilation_rate=[4, 8, 16], kernel_size = [10, 1], seq_len=height, width=width, num_classes = 2)
    
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        if ifslide:
            post_pred = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            post_label = postprocess(tf.argmax(y, 1), num_seg=num_seg, Threshold=0.5)
            correct = tf.equal(post_pred, post_label , name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        else:
            correct = tf.equal(predictions, tf.argmax(y, 1), name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        #accuracy_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(outputs, 1), tf.argmax(y, 1), num_classes, name='accuracy_per_class')

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)###,
    
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        np.random.seed(1998745)
        sess.run(iter.initializer)   # every trial restart training
        #sess.run(iter_test.initializer)    # every trial restart training
        sess.run(tf.global_variables_initializer())
        acc_total_train = np.array([])
        acc_total_test = np.array([])
        loss_total_train = np.array([])
        # track the outlier files
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        print results_dir
        for batch in range(total_batches):#####3
            save_name = results_dir + '/' + "_step{}_".format( batch)
            filename_train, labels_train =  sess.run(ele)   # names, 1s/0s
            #filename_test, labels_test =  sess.run(ele_test)   # names, 1s/0s

            data_train = []
            for ind in range(len(filename_train)):
                data = func.read_data(filename_train[ind], ifnorm=ifnorm)
                data_train.append(data)
            labels_train =  np.eye((num_classes))[labels_train.astype(int)]   # get one-hot lable
            #ipdb.set_trace()
            if ifslide:
                data_slide, labels_slide = func.sliding_window(np.array(data_train), labels_train, num_seg, window=seq_len/num_seg, stride=seq_len/num_seg )## 5s segment with 1s overlap
                data_train = data_slide
                labels_train = labels_slide
            #ipdb.set_trace()
            _, acc, c, pred, summary = sess.run([optimizer, accuracy, cost, predictions, summaries], feed_dict={x: data_train, y: labels_train})# , options=options, run_metadata=run_metadata We collect profiling infos for each step.
            writer.add_summary(summary, batch)

            if batch % test_every == 0:
                # track training
                acc_total_train = np.append(acc_total_train, acc)
                loss_total_train = np.append(loss_total_train, c)
                ###################### test ###################################### 
                data_file = np.load(data_dir_test)
                data_test, labels_test = data_file["x_test"], data_file["y_test"]
                test_accuracy = []
                
                for ii in range(labels_test.shape[0] // 50):   ## test with 100 per time and then average
                    data_test_batch = data_test[ii*50 : (ii+1)*50, :, :]
                    labels_test_batch = labels_test[ii*50:(ii+1)*50, :]
                    if ifslide:
                        data_slide_test, labels_slide_test = func.sliding_window(np.array(data_test_batch), labels_test_batch, num_seg, window=seq_len/num_seg, stride=seq_len/num_seg )## 5s segment with 1s overlap
                        data_test_batch = data_slide_test
                        labels_test_batch = labels_slide_test
                    test_temp = sess.run(accuracy, {x: data_test_batch, y: labels_test_batch})   # test_acc_sum, sensitivity_sum, specificity_sum,
                    test_accuracy.append(test_temp)
                test_accuracy = np.mean(test_accuracy)
                acc_total_test = np.append(acc_total_test, test_accuracy)
                ########################################################
            if batch % 10 == 0:
                print "batch:",batch, 'loss:', c, 'train-accuracy:', acc, 'test-accuracy:', test_accuracy
            if batch % save_every == 0:
                saver.save(sess, logdir + '/batch' + str(batch))

            if batch % plot_every == 0 and batch > test_every * smooth_win_len:   #

                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], window_len=smooth_win_len, xlabel= 'training batches / {}'.format(batch_size), ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_batch_{}".format(batch))

                func.plot_smooth_shadow_curve([loss_total_train], window_len=smooth_win_len, colors=['c'], xlabel= 'training batches / {}'.format(batch_size), ylabel="loss", title='Loss in training',labels=['training loss'], save_name=results_dir+ "/training_loss_batch_{}".format(batch))

                func.save_data((acc_total_train, loss_total_train, acc_total_test), header='accuracy_train,loss_train,accuracy_test', save_name=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO
        #np.savetxt('outliers.csv', outliers, fmt='%s', newline= ', ', delimiter=',')
    #coord.request_stop()
    #coord.join(threads)


if __name__ == "__main__":
    train(x)
### define the cost and opti
