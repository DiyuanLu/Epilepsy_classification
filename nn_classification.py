 ### use basic network to do classification
import numpy as np
#import matplotlib
#matplotlib.use('Agg')   ## when use cluster, do not output figures
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import functions as func
import modules as mod
import ipdb
from scipy.stats import ttest_ind
import pickle
from sklearn.model_selection import train_test_split
import argparse
import sys
from tqdm import tqdm
#from tensorflow.python.framework import ops
#ops.reset_default_graph()

#import json
#from sklearn.metrics import roc_auc_score

def get_arguments():
    def _str_to_bool(s):
        '''Convert string to bool (in argparse context).'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='EEG classification')

    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--logdir_root', type=str, default='/results',
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    return parser.parse_args()
    

datetime = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.datetime.now())
plot_every = 100
save_every = 2
test_every = 100
smooth_win_len = 20
ori_len = 10240
seq_len = 10240  #1280   ## 
start = 0
ifnorm = True
ifcrop = False   #True    ###
if ifcrop:
    crop_len = 9800
    seq_len = crop_len
        
width = 2  # with augmentation 2   ### data width
channels = 1
ifslide = False    ## True  #
if ifslide:
    ## 76 is from the correlation distribution result
    majority_vote = True
    post_process = 'majority_vote'   #'averaging_window'    ##
    window = 76
    stride = window // 5
    if ((ori_len - window) % stride) == 0:
        num_seg = (seq_len - window) // stride + 1
    else:
        num_seg = (seq_len - window) // stride
    ### use a 5s window slide over the 20s recording and do classification on segments, and then do a average vote
    height = window
    x = tf.placeholder('float32', [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
else:
            
    #seq_len = ori_len
    majority_vote = False
    post_process = 'None'  #'averaging_window'    ## 
    num_seg = 1
    height = seq_len
    x = tf.placeholder('float32', [None, height, width])
iffusion = False    ###True

if iffusion:
    post_process = None



y = tf.placeholder('float32')
learning_rate = tf.placeholder('float32')
#batch_size = tf.placeholder('int64')
print("num_seg", num_seg)

batch_size = 20  # old: 16     20has a very good result
num_classes = 2
epochs = 151

header = None
train_dir = 'data/Whole_data/train_data/'
test_dir = 'data/Whole_data/test_data/'
vali_dir = 'data/Whole_data/validate_data/'
pattern='Data*.csv'

#mod_params = './module_params.json'
#with open(mod_params, 'r') as f:
    #params = json.load(f)
model_name = 'CNN_Tutorial'  ##'ResNet'CNN_Tutorial
version = 'whole_{}_{}'.format(pattern[0:4], model_name)# AggResNet CNN_Tutorial CNN_Tutorial_Resi DeepConvLSTM   Atrous_CNN     PyramidPoolingConv  CNN_Tutorial       #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4]) Atrous_      #### DeepConvLSTMDeepCLSTMDilatedCNN
optimizer_name = "Adam"
#rand_seed = np.random.choice(200000)
rand_seed = 299
np.random.seed(rand_seed)
print('rand seed', rand_seed)


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

def evaluate_on_test(sess, epoch, accuracy, cost, confusion_matrix, outputs, activities=0, crop_len=10240, ifslide=False, ifnorm=True, ifcrop=False, header=None, save_name='results/'):
    acc_epoch_test = 0
    loss_epoch_test = 0
    confusion = 0
    data_dir = test_dir
    filename  = 'ori_test_data_label.npz'
    num_classes = 2
    
    try:
        data = np.load(data_dir + filename)
        data_test = data['data']
        labels_test = data['label'] 
    except:
        data_test, labels_test = func.load_and_save_data_to_npz(data_dir, pattern=pattern, withlabel=True, ifnorm=True, num_classes=num_classes, save_name=filename)

    labels_test_hot =  np.eye((num_classes))[labels_test.astype(int)]
    #ACT = np.zeros((len(labels_test_hot), 200))   ### collect fully connected activity
    ACT = []   ### collect fully connected activity
    ### randomly crop a crop_len
    if ifcrop:
        data_test = func.random_crop(data_test, crop_len=crop_len)
    ### add random noise
    #data_test = func.add_random_noise(data_test, prob=0.5, noise_amp=0.01)
                
    test_bs = 100
    logits = np.zeros((len(labels_test),))
    for jj in range(len(labels_test) // test_bs):
        if ifslide:
            data_slide = func.slide_and_segment(data_test[jj*test_bs: (jj+1)*test_bs, :, :], num_seg = num_seg, window=window, stride=stride)## 5s segment with 1s overlap
            data_test_batch = data_slide
            if not iffusion:
                labels_test_batch = np.repeat(labels_test_hot[jj*test_bs: (jj+1)*test_bs, :],  num_seg, axis=0).reshape(-1, labels_test_hot.shape[1])
            #labels_test_batch = labels_test_hot[jj*50: (jj+1)*50, :]
        else:
            data_test_batch, labels_test_batch  = data_test[jj*test_bs: (jj+1)*test_bs, :, :], labels_test_hot[jj*test_bs: (jj+1)*test_bs, :]

        test_acc, test_loss, logi, act = sess.run([accuracy, cost, outputs, activities], {x: data_test_batch, y: labels_test_batch, learning_rate:func.lr(epoch)})

        logits[jj*test_bs : (jj+1)*test_bs] = np.max(logi, axis=1)
        #ipdb.set_trace()
        #ACT[jj*test_bs : (jj+1)*test_bs, ...] = act
        #ACT.append(act)
        
        acc_epoch_test += test_acc
        loss_epoch_test += test_loss
        #confusion += conf
    #if acc_epoch_test /(jj + 1) > 0.85:
        #ipdb.set_trace()
        #func.plotTSNE(fc.astype(np.float64), np.argmax(labels_test_batch, axis=1), num_classes=num_classes, n_components=2, title="t-SNE", target_names = ['non-focal', 'focal'], save_name=save_name+'/fully-activity', postfix='t-SNE on Barceona dataset')
        #func.plotTSNE(fc.astype(np.float64), np.argmax(labels_test_batch, axis=1), num_classes=num_classes, n_components=3, title="t-SNE", target_names = ['non-focal', 'focal'], save_name=save_name+'/fully-activity', postfix='t-SNE on Barceona dataset')
    
    #func.plot_auc_curve(np.repeat(labels_test, num_seg, axis=0), logits, save_name=save_name+'/epoch_{}_test_'.format(epoch))
    ### input to this function is finally int label
    func.plot_auc_curve(labels_test, logits, save_name=save_name+'/epoch_{}_test_'.format(epoch))
    acc_epoch_test /= (jj + 1)
    loss_epoch_test /= (jj + 1)

    return acc_epoch_test, loss_epoch_test



    
### construct the network
def train(x):

        
    with tf.name_scope('Data'):
        #rand_seed = np.int(np.random.randint(0, 10000, 1))
        ### Get data. 
        files_wlabel = func.find_files(train_dir, pattern=pattern, withlabel=True)### traverse all the files in the dir, and divide into batches, from

        files, labels = np.array(files_wlabel)[:, 0].astype(np.str), np.array(np.array(files_wlabel)[:, 1]).astype(np.int)

        #### split into train and test
        num_train = len(labels)
        print('num_train', labels.shape)
        
        ### tensorflow dataset
        dataset_train = tf.data.Dataset.from_tensor_slices((files, labels)).shuffle(buffer_size=10000).repeat().batch(batch_size)
        iter = dataset_train.make_initializable_iterator()
        ele = iter.get_next() #you get the filename
            
    ################# Constructing the network ###########################
    #if model=='fc': outputs = mod.fc_net(x, hid_dims=[500, 300, 100], num_classes = num_classes)   ##
    #outputs, out_pre = mod.resi_net(x, hid_dims=[1500, 500], seq_len=height, width=width, channels=channels, num_blocks=5, num_classes = num_classes)
    #outputs = mod.CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], pool_size=[4, 1], strides=[4, 1], seq_len=height, width=width, channels=channels, num_classes = num_classes)
    #outputs = mod.CNN_new(x, output_channels=[4, 8, 16, 32], num_block=2, num_seg=num_seg, seq_len=height, width=width, channels=channels, num_classes = num_classes)    ## ok
    #outputs, kernels = mod.DeepConvLSTM(x, output_channels=[4, 8, 8], filter_size=[11, 1], pool_size=[4, 1], strides=[2, 1], num_lstm=64, group_size=4, seq_len=height, width=width, channels=channels, num_classes = num_classes)  ## ok
    #outputs = mod.RNN(x, num_lstm=128, seq_len=height, width=width, channels=channels, group_size=32, num_classes = num_classes)   ##ok
    #outputs = mod.Dilated_CNN(x, output_channels=16, seq_len=seq_len, width=width, channels=channels, num_classes = num_classes)
    #outputs = mod.Atrous_CNN(x, output_channels_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [5, 1], seq_len=height, width=width, channels=channels, num_classes = 2)
    #outputs = mod.PyramidPoolingConv(x, output_channels=[2, 4, 8, 16, 32], filter_size=7, dilation_rate=[2, 8, 16, 32], seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs = mod.Inception(x, filter_size=[5, 9],num_block=2, seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs = mod.Inception_complex(x, output_channels=[4, 8, 16, 32], filter_size=[5, 9], num_block=2, seq_len=height, width=width, channels=channels, num_classes=num_classes)
    #if model_name == 'ResNet': outputs, kernels = mod.ResNet(x, num_layer_per_block=3, filter_size=[[11, 1], [5, 1]], output_channels=[16, 32, 64], pool_size=[[4, 1]], strides=[4, 1], seq_len=height, width=width, channels=channels, num_classes=2)
    #if model_name == 'AggResNet': outputs, pre = mod.AggResNet(x, output_channels=[8, 16, 32], num_stacks=[3, 3, 3], cardinality=8, seq_len=height, width=width, channels=channels, filter_size=[9, 1], pool_size=[4, 1], strides=[4, 1], fc=[500], num_classes=num_classes)

    if model_name == 'CNN_Tutorial': outputs, kernels, activities = mod.CNN_Tutorial(x, output_channels=[8, 16, 32], seq_len=height, width=width, channels=channels, num_classes=num_classes, pool_size=[4, 1], strides=[4, 1], filter_size=[[9, 1], [5, 1]], fc=[500, 200], iffusion=iffusion, num_seg=num_seg) ## works on CIFAR, for BB pool_size=[4, 1], strides=[4, 1], filter_size=[9, 1], fc1=200 works well.
    #if model_name == 'CNN_Tutorial_Resi': outputs, fc_act = mod.CNN_Tutorial_Resi(x, output_channels=[8, 16, 32, 64], seq_len=height, width=width, channels=1, pool_size=[4, 1], strides=[4, 1], filter_size=[[9, 1], [5, 1]], num_classes=num_classes, fc=[200])
    #if model_name == 'RNN_Tutorial': outputs, kernels = mod.RNN_Tutorial(x, num_rnn=[50, 50], seq_len=height, width=width, channels=channels, fc=[50, 50], drop_rate=0.5, group_size=1, num_classes = num_classes)
    #if model_name == 'CNN_Tutorial_attention': outputs, kernels, activities, diversity = mod.CNN_Tutorial_attention(x, output_channels=[8, 16, 32], seq_len=height, width=width, channels=channels, pool_size=[4, 1], strides=[4, 1], filter_size=[5, 1], num_att=5, num_classes=num_classes, fc=[200], gird_height=5, gird_width=1, att_dim=128)
    #ipdb.set_trace()
    #### specify logdir
    results_dir= 'results/' + version + '/cpu-batch{}/seg_len{}-conv16-32-32-p4-s4-f9-f5-fc200-lr0.01-{}-'.format(batch_size, seq_len, optimizer_name)+ datetime
    #cnv4_lstm64testcrop10000-add-noise-CNN-dropout0.3-'.format(batch_size, num_seg, majority_vote)
    ##seg_len166-conv5,3-p4-s3-conv8-16-32-fc100-
    logdir = results_dir+ '/model'

    ### Load model if specify
    args = get_arguments()
    
    if not args.restore_from:
        restore_from = logdir
    else:
        restore_from = args.restore_from

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from
    

    with tf.name_scope('loss'):
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name='cost')
        if model_name == 'CNN_Tutorial_attention':
            reg = tf.sqrt(diversity)
            reg = tf.matmul(reg, tf.transpose(reg, [0, 2, 1]))
            #ipdb.set_trace()
            reg = reg-tf.eye(reg.shape.as_list()[1])
            reg = tf.pow(reg, 2)
            reg = tf.reduce_sum(reg) + 1e-5
            reg = tf.sqrt(reg)
            reg = tf.reduce_mean(reg)
            lost = cost + reg
            tf.summary.scalar('diversity', reg)
            tf.summary.scalar('loss', lost)
    with tf.name_scope('performance'):
        predictions = tf.argmax(outputs, 1)
        if post_process == 'majority_vote':
            #ipdb.set_trace()
            predictions = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            labels = postprocess(tf.argmax(y, 1), num_seg=num_seg, Threshold=0.5)
            correct = tf.equal(predictions, labels , name='correct')
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'), name='accuracy')
        elif post_process == 'averaging_window':
            predictions = postprocess(predictions, num_seg=num_seg, Threshold=0.5)
            labels = average_window(tf.argmax(y, 1), window=4, threshold=0.6)
            correct = tf.equal(predictions, labels, name='correct')
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'), name='accuracy')
        else:
            correct = tf.equal(predictions, tf.argmax(y, 1), name='correct')##
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'), name='accuracy')

        ### auc always 0.0, https://stackoverflow.com/questions/49887325/tensorflow-always-getting-an-auc-value-of-0
        area_under_curve = tf.contrib.metrics.streaming_auc(labels=y, predictions=outputs, name='auc')[1]
        confusion_matrix = tf.confusion_matrix(labels, predictions, num_classes=num_classes, name='confusion')

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('auc', area_under_curve)
        
    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                    beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(cost)###,
    #if optimizer_name == 'SGD':
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                    #beta1=0.9,
                                   #beta2=0.999,
                                   #epsilon=1e-08).minimize(cost)###,
    #if optimizer_name == 'RMS':
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                            #decay=0.9,
                                            #momentum=0.999,
                                            #epsilon=1e-10).minimize(cost)   ### laerning rate 0.01 works
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    best_saver = tf.train.Saver(max_to_keep= 3)

    with tf.Session() as sess:
        try:
            #ipdb.set_trace()
            saved_global_step = func.load_model(best_saver, sess, restore_from)
            if is_overwritten_training or saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1

        except:
            print('Something went wrong while restoring checkpoint. '
                  'We will terminate training to avoid accidentally overwriting '
                  'the previous model.')
            raise
        tf.set_random_seed(rand_seed)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        sess.run(iter.initializer) # every trial restart training
        print('random number', np.random.randint(0, 50, 10))
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
        best_eval_acc = 0.0
        for epoch in range(epochs):
            acc_epoch_train = 0
            loss_epoch_train = 0
            for batch in tqdm(range(num_train//batch_size)):#####                
                save_name = results_dir + '/' + 'step{}_'.format( batch)
                
                filename_train, labels_train =  sess.run(ele)   # names, 1s/0s the filename is bytes object!!! TODO
                #print("epoch-", epoch, 'batch', batch, filename_train)
                data_train = np.zeros([batch_size, ori_len, width])
                filename_train = filename_train.astype(np.str)

                for ind in range(len(filename_train)):
                    data = func.read_data(filename_train[ind],  header=header, ifnorm=True, start=start, width=width)
                    data_train[ind, :, :] = data
                    
                # data augmentation
                ## randomly crop a target_len
                if ifcrop:
                    data_train = func.random_crop(data_train, crop_len=crop_len)

                ### add random noise
                #data_train = func.add_random_noise(data_train, prob=0.5, noise_amp=0.01)

                labels_train_hot =  np.eye((num_classes))[labels_train.astype(int)] # get one-hot lable

                #if epoch == 0 and batch == 0:
                    #func.plot_BB_training_examples(data_train[0:10, :, :], labels_train[0:10], save_name=save_name)
                #ipdb.set_trace()
                if ifslide:
                    data_slide = func.slide_and_segment(data_train, num_seg=num_seg, window=window, stride=stride )## 5s segment with 1s overlap
                    data_train = data_slide
                    if not iffusion:
                        labels_train_hot = np.repeat(labels_train_hot, num_seg, axis=0).reshape(-1, labels_train_hot.shape[1])
                #ipdb.set_trace()
                _, summary, acc, c = sess.run([optimizer, summaries, accuracy, cost], feed_dict={x: data_train, y: labels_train_hot, learning_rate:func.lr(epoch)})# , options=options, run_metadata=run_metadata We collect profiling infos for each step.
                writer.add_summary(summary, epoch*(num_train//batch_size)+batch)##
                
                ## accumulate the acc and cost later to average
                acc_epoch_train += acc
                loss_epoch_train += c
                #if acc_epoch_train / (batch+1) > 0.91:
                    #ipdb.set_trace()
                    #act_train, divers = sess.run([activities, diversity], feed_dict={x: data_train, y: labels_train_hot, learning_rate:func.lr(epoch)})
                    #func.plotTSNE(fc.astype(np.float64), np.argmax(labels_train_hot, axis=1), num_classes=num_classes, n_components=2, title="t-SNE", target_names = ['non-focal', 'focal'], save_name=save_name+'/fully-activity', postfix='t-SNE on Barceona dataset')
                
            print('epoch', epoch, 'train-accuracy:', acc_epoch_train / batch)
            ###################### test ######################################
            if epoch % 1 == 0:                                
                acc_epoch_test, loss_epoch_test = evaluate_on_test(sess, epoch, accuracy, cost, confusion_matrix, outputs, activities=activities, ifslide=ifslide, ifnorm=ifnorm, ifcrop=ifcrop, header=header, save_name=results_dir)
                print('epoch', epoch, '\ntest-accuracy:', acc_epoch_test)
            ########################################################
                
            # track training and testing
            loss_total_train.append(loss_epoch_train / (batch + 1))            
            acc_total_train.append(acc_epoch_train / (batch + 1))
            loss_total_test.append(loss_epoch_test)            
            acc_total_test.append(acc_epoch_test)
            
            if acc_epoch_test >= best_eval_acc:
                print("Found new best accuracy {}".format(acc_epoch_test))
                func.save_model(best_saver, sess, logdir, epoch, best_eval_acc)
                last_saved_step = epoch
                best_eval_acc = acc_epoch_test

            if epoch % 1 == 0:
                
                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], ifsmooth=False, hlines=[0.8, 0.85, 0.9], window_len=smooth_win_len, xlabel= 'training epochs', ylabel='accuracy', colors=['darkcyan', 'm'], ylim=[0.45, 1.05], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ '/learning_curve_epoch_{}_seed{}'.format(epoch, rand_seed))

                func.plot_smooth_shadow_curve([loss_total_train, loss_total_test], window_len=smooth_win_len, ifsmooth=False, hlines=[], colors=['c', 'violet'], ylim=[0.05, 0.9], xlabel= 'training epochs', ylabel='loss', title='Loss',labels=['training loss', 'test loss'], save_name=results_dir+ '/loss_epoch_{}_seed{}'.format(epoch, rand_seed))

                func.save_data_to_csv((np.array(acc_total_train), np.array(loss_total_train), np.array(acc_total_test), np.array(loss_total_test)), header='accuracy_train,loss_train,accuracy_test,loss_test', save_name=results_dir + '/' + datetime + 'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO

    ##Stop the threads
    #coord.request_stop()
    
    ##Wait for threads to stop
    #coord.join(threads)


if __name__ == '__main__':
    train(x)
### define the cost and opti
