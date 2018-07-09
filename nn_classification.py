 ### use basic network to do classification
import numpy as np
import matplotlib
matplotlib.use('Agg')   ## when use cluster, do not output figures
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
#from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import modules as mod
from sklearn.model_selection import KFold, StratifiedKFold
import ipdb
from scipy.stats import ttest_ind
#import cPickle as pickle
import pickle


kfolds = 10
skf = StratifiedKFold(n_splits=kfolds, shuffle=True)   ## keep the class ratio balance in each fold

def lr(epoch):
    learning_rate = 0.0005
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 20:
        learning_rate *= 1e-3
    elif epoch > 10:
        learning_rate *= 1e-2
    elif epoch > 4:
        learning_rate *= 1e-1
    return learning_rate

#ipdb.set_trace()
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 100
save_every = 2
test_every = 50
smooth_win_len = 20
seq_len = 10240  #1280   ## 
height = seq_len
width = 2  # with augmentation 2   ### data width
channels = 1
start = 0
ifnorm = True
ifslide = False    #True  ##
if ifslide:
    majority_vote = True
    num_seg = 5   ## number of shorter segments you want to divide the original long sequence with a sliding window
    ### use a 5s window slide over the 20s recording and do classification on segments, and then do a average vote
    height = np.int(height / num_seg)
    x = tf.placeholder("float32", [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
else:
    majority_vote = False
    num_seg = 1
    x = tf.placeholder("float32", [None, height, width])
y = tf.placeholder("float32")
learning_rate = tf.placeholder("float32")

post_process = "majority_vote"   #'averaging_window'    ## 
batch_size = 20  # old: 16     20has a very good result
num_classes = 2
epochs = 200
             #
header = None
data_dir = "data/train_data"
pattern='Data*.csv'
data_version = 'Data'
version = 'whole_{}_CNN_Tutorial'.format(pattern[0:4])# AggResNet   DeepConvLSTM   Atrous_CNN     PyramidPoolingConv         #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4]) Atrous_      #### DeepConvLSTMDeepCLSTMDilatedCNN
results_dir= "results/" + version + '/cpu-batch{}/slide{}-vote{}'.format(batch_size, num_seg, majority_vote)+ datetime#cnv4_lstm64test

logdir = results_dir+ "/model"


def postprocess(prediction, num_seg=10, Threshold=0.5):
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


def average_window(prediction, window=4, threshold=0.6):
    '''sliding averaging window'''

    filters = np.ones(window) / window
    result = np.convolve(prediction, filters, 'same')

    return result


    
def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,784],order='F'),keep_prob:1.0})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")


### construct the network
def train(x):

    with tf.name_scope("Data"):
        ### Get data. In each file, the data_len is different. But training, split them into 4s segments and do classification on the segements
        files_wlabel = func.find_files(data_dir, pattern=pattern, withlabel=True)### traverse all the files in the dir, and divide into batches, from
        #ipdb.set_trace()
        print("files_wlabel", files_wlabel[0])

        files, labels = np.array(files_wlabel)[:, 0].astype(np.str), np.array(np.array(files_wlabel)[:, 1]).astype(np.int)

        #### split into train and test
        # ipdb.set_trace()
        skf.get_n_splits(files, labels)
        for train_index, test_index in skf.split(files, labels):
            files_train, files_test = files[train_index], files[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
        num_test = len(test_index)
        num_train = len(files_train)
        print("num_train", num_train, "num_test", num_test)
        
        ### tensorflow dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        iter = dataset_train.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        

            
    ################# Constructing the network ###########################
    #outputs = mod.fc_net(x, hid_dims=[500, 300, 100], num_classes = num_classes)   ##
    #outputs = mod.resi_net(x, hid_dims=[500, 300], seq_len=height, width=width, channels=channels, num_blocks=2, num_classes = num_classes)
    #outputs = mod.CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], pool_size=[4, 1], strides=[4, 1], seq_len=height, width=width, channels=channels, num_classes = num_classes)
    #outputs = mod.CNN_new(x, output_channels=[4, 8, 16, 32], num_block=2, num_seg=num_seg, seq_len=height, width=width, channels=channels, num_classes = num_classes)    ## ok
    #outputs = mod.DeepConvLSTM(x, output_channels=[8, 16, 32], filter_size=[9, 1], pool_size=[4, 1], strides=[4, 1], num_lstm=64, group_size=8, seq_len=height, width=width, channels=channels, num_classes = num_classes)  ## ok
    #outputs = mod.RNN(x, num_lstm=128, seq_len=height, width=width, channels=channels, group_size=32, num_classes = num_classes)   ##ok
    #outputs = mod.Dilated_CNN(x, output_channels=16, seq_len=seq_len, width=width, channels=channels, num_classes = num_classes)
    #outputs = mod.Atrous_CNN(x, output_channels_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [5, 1], seq_len=height, width=width, channels=channels, num_classes = 2)
    #outputs = mod.PyramidPoolingConv(x, output_channels=[2, 4, 8, 16, 32], filter_size=7, dilation_rate=[2, 8, 16, 32], seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs = mod.Inception(x, filter_size=[5, 9],num_block=2, seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs = mod.Inception_complex(x, output_channels=[4, 8, 16, 32], filter_size=[5, 9], num_block=2, seq_len=height, width=width, channels=channels, num_classes=num_classes)
    #outputs = mod.ResNet(x, num_layer_per_block=3, num_block=4, output_channels=[20, 32, 64, 128], seq_len=height, width=width, channels=channels, num_classes=2)
    #outputs = mod.AggResNet(x, output_channels=[4, 8, 16], num_stacks=[3, 3, 3], cardinality=16, seq_len=height, width=width, channels=channels, filter_size=[7, 1], pool_size=[4, 1], strides=[2, 1], fc=500, num_classes=num_classes)

    outputs = mod.CNN_Tutorial(x, output_channels=[8, 16, 32], seq_len=height, width=width, channels=channels, num_classes=num_classes, pool_size=[4, 1], strides=[4, 1], filter_size=[[9, 1], [5, 1]], fc1=1500) ## works on CIFAR, for BB pool_size=[4, 1], strides=[4, 1], filter_size=[9, 1], fc1=200 works well.
    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        if post_process == 'majority_vote':
            #ipdb.set_trace()
            post_pred = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            post_label = postprocess(tf.argmax(y, 1), num_seg=num_seg, Threshold=0.5)
            correct = tf.equal(post_pred, post_label , name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        if post_process == 'averaging_window':
            post_pred = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            post_label = average_window(predictions, window=4, threshold=0.6)
            correct = tf.equal(post_pred, post_label , name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        else:
            correct = tf.equal(predictions, tf.argmax(y, 1), name="correct")##
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        #accuracy_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(outputs, 1), tf.argmax(y, 1), num_classes, name='accuracy_per_class')

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                    beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(cost)###,
    #optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)   ### laerning rate 0.01 works
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.Session() as sess:
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        np.random.seed(1998745)
        sess.run(iter.initializer)   # every trial restart training
        sess.run(tf.global_variables_initializer())
        acc_total_train = []
        acc_total_test = []
        loss_total_train = []
        loss_total_test = []
        # track the outlier files
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        print(results_dir)
        # ipdb.set_trace()
        for epoch in range(epochs):
            acc_epoch_train = 0
            loss_epoch_train = 0
            for batch in range(num_train//batch_size):#####
                save_name = results_dir + '/' + "step{}_".format( batch)
                filename_train, labels_train =  sess.run(ele)   # names, 1s/0s the filename is bytes object!!! TODO
                data_train = np.zeros([batch_size, seq_len, width])
                filename_train = filename_train.astype(np.str)
                for ind in range(len(filename_train)):
                    data = func.read_data(filename_train[ind],  header=header, ifnorm=True, start=start, width=width)
                    data_train[ind, :, :] = data

                labels_train_hot =  np.eye((num_classes))[labels_train.astype(int)] # get one-hot lable

                if epoch == 0 and batch == 0:
                    func.plot_BB_training_examples(data_train[0:10, :, :], labels_train[0:10], save_name=save_name)
                # ipdb.set_trace()
                if ifslide:
                    data_slide = func.slide_and_segment(data_train, num_seg, window=seq_len//num_seg, stride=seq_len//num_seg )## 5s segment with 1s overlap
                    data_train = data_slide
                    labels_train_hot = np.repeat(labels_train_hot,  num_seg, axis=0).reshape(-1, labels_train_hot.shape[1])
                #ipdb.set_trace()
                _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: data_train, y: labels_train_hot, learning_rate:lr(epoch)})# , options=options, run_metadata=run_metadata We collect profiling infos for each step.
                writer.add_summary(summary, batch)


                ## accumulate the acc and cost later to average
                acc_epoch_train += acc
                loss_epoch_train += c
                ###################### test ######################################
                if batch % test_every == 0:
                    # track test
                    
                    acc_epoch_test = 0
                    loss_epoch_test = 0
                    data_test_tot = np.zeros((num_test, seq_len, width))
                    # for ii in range(labels_test.shape[0]):   ## test with 100 per time and then average
                    for ind, filename in enumerate( files_test):
                        data = func.read_data(filename, header=header, ifnorm=True, start=start, width=width)
                        data_test_tot[ind, :, :] = data
                    labels_test_hot =  np.eye((num_classes))[labels_test.astype(int)]
                    
                    for jj in range(num_test // 50):
                        if ifslide:
                            data_slide = func.slide_and_segment(data_test_tot[jj*50: (jj+1)*50, :, :], num_seg, window=seq_len//num_seg, stride=seq_len//num_seg )## 5s segment with 1s overlap
                            data_test_batch = data_slide
                            labels_test_batch = np.repeat(labels_test_hot[jj*50: (jj+1)*50, :],  num_seg, axis=0).reshape(-1, labels_test_hot.shape[1])
                            #labels_test_batch = labels_test_hot[jj*50: (jj+1)*50, :]
                        else:
                            data_test_batch, labels_test_batch  = data_test_tot[jj*50: (jj+1)*50, :, :], labels_test_hot[jj*50: (jj+1)*50, :]

                        test_acc, test_pred, test_loss = sess.run([accuracy, outputs, cost], {x: data_test_batch, y: labels_test_batch, learning_rate:lr(epoch)})
                        
                        acc_epoch_test += test_acc
                        loss_epoch_test += test_loss
                        
                    acc_epoch_test /= (jj + 1)
                    loss_epoch_test /= (jj + 1)
                    
                    print('epoch', epoch, "batch:",batch, 'loss:', c, 'train-accuracy:', acc, 'test-accuracy:', acc_epoch_test)
                    ########################################################
            # track training and testing
            loss_total_train.append(loss_epoch_train / (batch + 1))            
            acc_total_train.append(acc_epoch_train / (batch + 1))
            loss_total_test.append(loss_epoch_test)            
            acc_total_test.append(acc_epoch_test)
            
            if epoch % save_every == 0:
                    saver.save(sess, logdir + '/epoch' + str(epoch))                   
            
            if epoch % 1 == 0:
                
                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], ifsmooth=False, window_len=smooth_win_len, xlabel= 'training epochs', ylabel="accuracy", colors=['darkcyan', 'm'], ylim=[0.45, 1.05], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_epoch_{}".format(epoch))

                func.plot_smooth_shadow_curve([loss_total_train, loss_total_test], window_len=smooth_win_len, ifsmooth=False, colors=['c', 'violet'], ylim=[0.05, 0.9], xlabel= 'training epochs', ylabel="loss", title='Loss',labels=['training loss', 'test loss'], save_name=results_dir+ "/loss_epoch_{}".format(epoch))

                func.save_data((acc_total_train, loss_total_train, acc_total_test, loss_total_test), header='accuracy_train,loss_train,accuracy_test,loss_test', save_name=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO

        #np.savetxt('outliers.csv', outliers, fmt='%s', newline= ', ', delimiter=',')
    #coord.request_stop()
    #coord.join(threads)


if __name__ == "__main__":
    train(x)
### define the cost and opti
