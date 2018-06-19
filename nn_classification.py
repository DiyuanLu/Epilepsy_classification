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

kfolds = 10
skf = StratifiedKFold(n_splits=kfolds, shuffle=True)   ## keep the class ratio balance in each fold


data_dir = "data/train_data"
  #ori_50train20test.npz"###ori_aug2_20tes

#ipdb.set_trace()
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 200
save_every = 200
test_every = 10
smooth_win_len = 20
seq_len = 10240  #1280   ## 
height = seq_len
width = 2  # with augmentation 2   ### data width
num_seg = 5   ## number of shorter segments you want to divide the original long sequence with a sliding window
ifnorm = True
ifslide = True  ##False    #   
majority_vote = True   #False
batch_size = 30  # old: 16     20has a very good result
num_classes = 2
epochs = 50
total_batches =  epochs * 6000 // batch_size + 1 #5001               #


pattern='Data*.csv'
version = 'whole_{}_CNN'.format(pattern[0:4])#    DeepConvLSTM   Atrous_CNN     PyramidPoolingConv         #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4]) Atrous_      #### DeepConvLSTMDeepCLSTMDilatedCNN
results_dir= "results/" + version + '/cpu-batch{}/2block-' .format(batch_size)+ datetime#cnv4_lstm64test
logdir = results_dir+ "/model"

if ifslide:   ### use a 5s window slide over the 20s recording and do classification on segments, and then do a average vote
    height = np.int(height / num_seg)
    x = tf.placeholder("float32", [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
else:
    x = tf.placeholder("float32", [None, height, width])
y = tf.placeholder("float32")

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
        with tf.name_scope("Data"):
            ### Get data. In each file, the data_len is different. But training, split them into 4s segments and do classification on the segements
            files_wlabel = func.find_files(data_dir, pattern=pattern, withlabel=True)### traverse all the files in the dir, and divide into batches, from
            #ipdb.set_trace()
            print("files_wlabel", files_wlabel[0])

            files, labels = np.array(files_wlabel)[:, 0].astype(np.str), np.array(np.array(files_wlabel)[:, 1]).astype(np.int)
            # split into train and test
            # ipdb.set_trace()
            skf.get_n_splits(files, labels)
            for train_index, test_index in skf.split(files, labels):
                files_train, files_test = files[train_index], files[test_index]
                labels_train, labels_test = labels[train_index], labels[test_index]
            num_test = len(test_index)
            dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
            iter = dataset_train.make_initializable_iterator()
            ele = iter.get_next()   #you get the filename

    ################# Constructing the network ###########################
    #outputs = mod.fc_net(x, hid_dims=[500, 300, 100], num_classes = num_classes)   ##
    ##outputs = mod.resi_net(x, hid_dims=[500, 300], num_classes = num_classes)  ## ok very sfast
    outputs = mod.CNN(x, num_filters=[2, 4, 8, 16], num_block=2, filter_size=9, seq_len=height, width=width, num_classes = num_classes)    ## ok
    #outputs = mod.CNN_new(x, num_filters=[4, 8, 16, 32], num_block=3, num_seg=num_seg, seq_len=height, width=width, num_classes = num_classes)    ## ok
    #outputs = mod.DeepConvLSTM(x, num_filters=[8, 16, 32], filter_size=9, num_lstm=64, seq_len=height, width=width, num_classes = num_classes)  ## ok
    #outputs = mod.RNN(x, num_lstm=128, seq_len=height, width=width, group_size=16, num_classes = num_classes)   ##ok
    #outputs = mod.Dilated_CNN(x, num_filters=16, seq_len=seq_len, width=width, num_classes = num_classes)
    #outputs = mod.Atrous_CNN(x, num_filters_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [5, 1], seq_len=height, width=width, num_classes = 2)
    #outputs = mod.PyramidPoolingConv(x, num_filters=[2, 4, 8, 16, 32], filter_size=7, dilation_rate=[2, 8, 16, 32], seq_len=height, width=width, num_seg=num_seg, num_classes=num_classes)

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        if majority_vote:
            #ipdb.set_trace()
            post_pred = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            post_label = postprocess(tf.argmax(y, 1), num_seg=num_seg, Threshold=0.5)
            correct = tf.equal(post_pred, post_label , name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        else:
            correct = tf.equal(predictions, tf.argmax(y, 1), name="correct")##
            accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")
        #accuracy_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(outputs, 1), tf.argmax(y, 1), num_classes, name='accuracy_per_class')

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)

    #optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)###,
    optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
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
        print(results_dir)
        # ipdb.set_trace()
        for batch in range(total_batches):#####
            save_name = results_dir + '/' + "step{}_".format( batch)
            filename_train, labels_train =  sess.run(ele)   # names, 1s/0s the filename is bytes object!!! TODO
            data_train = np.zeros([batch_size, seq_len, width])
            filename_train = filename_train.astype(np.str)
            for ind in range(len(filename_train)):
                # ipdb.set_trace()
                # print("filename_train[ind]", filename_train[ind])
                data = func.read_data(filename_train[ind],  header=None, ifnorm=False)
                data_train[ind, :, :] = data
            labels_train_hot =  np.eye((num_classes))[labels_train.astype(int)]   # get one-hot lable
            # ipdb.set_trace()
            if batch == 0:
                #ipdb.set_trace()
                func.plot_BB_training_examples(data_train, labels_train, save_name=save_name)

            # ipdb.set_trace()
            if ifslide:
                data_slide = func.slide_and_segment(data_train, num_seg, window=seq_len//num_seg, stride=seq_len//num_seg )## 5s segment with 1s overlap
                data_train = data_slide
                #labels_train_hot = np.repeat(labels_train_hot,  num_seg, axis=0).reshape(-1, labels_train_hot.shape[1])

            _, acc, c, summary = sess.run([optimizer, accuracy, cost, summaries], feed_dict={x: data_train, y: labels_train_hot})# , options=options, run_metadata=run_metadata We collect profiling infos for each step.
            writer.add_summary(summary, batch)

            ###################### test ######################################
            if batch % test_every == 0:
                # track training
                acc_total_train = np.append(acc_total_train, acc)
                loss_total_train = np.append(loss_total_train, c)
                data_test_tot = np.zeros((num_test, seq_len, width))
                # for ii in range(labels_test.shape[0]):   ## test with 100 per time and then average
                for ind, filename in enumerate( files_test):
                    data = func.read_data(filename, header=None, ifnorm=False)
                    data_test_tot[ind, :, :] = data
                labels_test_hot =  np.eye((num_classes))[labels_test.astype(int)]
                
                test_acc_batch = 0
                for jj in range(num_test // 50):
                    if ifslide:
                        data_slide = func.slide_and_segment(data_test_tot[jj*50: (jj+1)*50, :, :], num_seg, window=seq_len//num_seg, stride=seq_len//num_seg )## 5s segment with 1s overlap
                        data_test_batch = data_slide
                        #labels_test_batch = np.repeat(labels_test_hot,  num_seg, axis=0).reshape(-1, labels_test_hot.shape[1])
                        labels_test_batch = labels_test_hot[jj*50: (jj+1)*50, :]
                    else:
                        data_test_batch, labels_test_batch  = data_test_tot[jj*50: (jj+1)*50, :, :], labels_test_hot[jj*50: (jj+1)*50, :]
                    
                    test_temp, test_pred = sess.run([accuracy, outputs], {x: data_test_batch, y: labels_test_batch})   
                    test_acc_batch += test_temp
                test_accuracy = test_acc_batch / (num_test // 50)
                acc_total_test = np.append(acc_total_test, test_accuracy)
                print("batch:",batch, 'loss:', c, 'train-accuracy:', acc, 'test-accuracy:', test_accuracy)
                ########################################################

            if batch % save_every == 0:
                saver.save(sess, logdir + '/batch' + str(batch))

            if batch % plot_every == 0 and batch > test_every * smooth_win_len:   #

                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], ifsmooth=True, window_len=smooth_win_len, xlabel= 'training batches / {}'.format( test_every), ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_batch_{}".format(batch))

                func.plot_smooth_shadow_curve([loss_total_train], window_len=smooth_win_len, ifsmooth=True, colors=['c'], xlabel= 'training batches / {}'.format( test_every), ylabel="loss", title='Loss in training',labels=['training loss'], save_name=results_dir+ "/training_loss_batch_{}".format(batch))

                func.save_data((acc_total_train, loss_total_train, acc_total_test), header='accuracy_train,loss_train,accuracy_test', save_name=results_dir + '/' +'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO

        #np.savetxt('outliers.csv', outliers, fmt='%s', newline= ', ', delimiter=',')
    #coord.request_stop()
    #coord.join(threads)


if __name__ == "__main__":
    train(x)
### define the cost and opti
