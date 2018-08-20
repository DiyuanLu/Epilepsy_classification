 ### use basic network to do classification
import numpy as np
import matplotlib
matplotlib.use('Agg')   ## when use cluster, do not output figures
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
#import json
import matplotlib.gridspec as gridspec
#import pandas as pd
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
ori_len = 4097  #1280   ## 
seq_len = 4097  #1280   ## 
ifnorm = True
width = 1  # with augmentation 2   ### data width
channels = 1
ifcrop = True   ##False   # 
if ifcrop:
    crop_len = 3800
    seq_len = crop_len
else:
    crop_len = ori_len
    
ifslide = False    ##True  #
if ifslide:
    ## 76 is from the correlation distribution result
    majority_vote = True
    post_process = 'majority_vote'   #'averaging_window'    ##
    window = 80
    stride = window // 5
    if ((seq_len - window) % stride) == 0:
        num_seg = (seq_len - window) // stride + 1
    else:
        num_seg = (seq_len - window) // stride
    ### use a 5s window slide over the 20s recording and do classification on segments, and then do a average vote
    height = window
    x = tf.placeholder('float32', [None, height, width])  #20s recording width of each recording is 1, there are 2 channels
else:
    majority_vote = False
    post_process = 'None'  #'averaging_window'    ## 
    num_seg = 1
    height = seq_len
    x = tf.placeholder('float32', [None, height, width])
print("num_seg", num_seg)
y = tf.placeholder('float32')
learning_rate = tf.placeholder('float32')




batch_size = 8  # old: 16     20has a very good result
num_classes = 3
epochs = 151
header = None

train_dir = 'data/Bonn_data/'
#test_dir = 'data/Whole_data/test_data/'
#vali_dir = 'data/Whole_data/validate_data/'
pattern='*.csv'

#mod_params = './module_params.json'
#with open(mod_params, 'r') as f:
    #params = json.load(f)
model_name = 'CNN_Tutorial'   ##'ResNet'   ###
version = 'Bonn_{}'.format( model_name)
#CNN_Tutorial  CNN_Tutorial CNN_Tutorial_Resi DeepConvLSTM   Atrous_CNN     PyramidPoolingConv  CNN_Tutorial       #DeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4]) Atrous_      #### DeepConvLSTMDeepCLSTMDilatedCNN

#rand_seed = np.random.choice(200000)
rand_seed = 140376
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


                    
def evaluate_on_test(sess, epoch, accuracy, cost, outputs, test_data, kernels, activities=0, crop_len=10000, ifslide=False, ifnorm=True, ifcrop=False, header=None, save_name='results/'):
    
    acc_epoch_test = 0
    loss_epoch_test = 0
    labels_test, data_test = test_data[:, 0], test_data[:, 1:]
    labels_test_hot =  np.eye((num_classes))[labels_test.astype(int)]
    data_test = np.expand_dims(data_test, 2)
    ### randomly crop a crop_len
    if ifcrop:
        data_test = func.random_crop(data_test, crop_len=crop_len)
    ### add random noise
    #data_test = add_random_noise(data_test, prob=0.5, noise_amp=0.01)
                
    logits = []

    if ifslide:
        data_slide = func.slide_and_segment(data_test, num_seg=num_seg, window=window, stride=stride)## 5s segment with 1s overlap
        data_test_batch = data_slide
        labels_test_batch = np.repeat(labels_test_hot, num_seg, axis=0).reshape(-1, labels_test_hot.shape[1])
        #labels_test_batch = labels_test_hot[jj*50: (jj+1)*50, :]
    else:
        data_test_batch, labels_test_batch  = data_test, labels_test_hot
        
    #test_acc, test_loss, logi, train_vars, act = sess.run([accuracy, cost, outputs, kernels, activities], {x: data_test_batch, y: labels_test_batch, learning_rate:func.lr(epoch)})
    test_acc, test_loss, logi, train_vars, act = sess.run([accuracy, cost, outputs, kernels, activities], {x: data_test_batch, y: labels_test_batch, learning_rate:func.lr(epoch)})
    ###
    logits = np.max(logi, axis=1)

    
    #if epoch % 5 == 0:
        #for ind, var in enumerate(train_vars):
            #if 'fully' in var:
                #if train_vars[var].shape[-1] > num_classes:
                    #plt.imshow(train_vars[var], cmap='viridis', aspect='auto')
                    #plt.title(var + '-' + np.str(train_vars[var].shape))
                    #plt.ylabel('in unit index')
                    #plt.xlabel('out unit index')
                    #plt.colorbar()
                    #plt.savefig(save_name + '/fully-hid-' + np.str(train_vars[var].shape)+'-epoch-{}.png'.format(epoch), format='png')
                    #plt.close()
                #else:
                    
                    #for ind in range(train_vars[var].shape[-1]):
                        #plt.plot(train_vars[var][:, ind], label='label-{}'.format(ind))
                    ##plt.imshow(train_vars[var], cmap='viridis', aspect='auto')
                    ##plt.legend(loc='best')
                    #plt.ylabel('out unit index')
                    #plt.xlabel('in unit index')
                    #plt.savefig(save_name + '/fully-logits-imshow' + np.str(train_vars[var].shape)+'-epoch-{}.png'.format(epoch), format='png')
                    #plt.close()
            #if 'conv1' in var:
                #for ind in range(train_vars[var].shape[3]):
                    #ax = plt.subplot(8, 8, ind+1)
                    #plt.plot(train_vars[var][:, 0, 0, ind], 'slateblue', label='kernel-{}'.format(ind))
                    ##plt.legend()
                    #plt.setp(ax.get_xticklabels(), visible = False)
                    #plt.setp(ax.get_yticklabels(), visible = False)
                #plt.savefig(save_name + '/line-plot-filterse-conv1' + np.str(train_vars[var].shape)+'-epoch-{}.png'.format(epoch), format='png')
                #plt.close()
                #ipdb.set_trace()
        ### go through all the recorded layer activations
    
    if test_acc > 0.98:
        if epoch == 0:
            func.plotTSNE(np.reshape(data_test_batch, [len(labels_test), -1]).astype(np.float64), labels_test, num_classes=num_classes, n_components=2, title="t-SNE ", target_names=['healthy', 'unhealthy', 'seizure'], save_name=save_name+'/Original_data_tsne-epoch{}-'.format(epoch), postfix='on Bonn dataset')
        for ind, activity in enumerate(act):
            name = activity[0:5]
            num_sample = 1     ## each class plot 3 samples
            ### for each label plot 2 examples of all activations
            for label in range(num_classes):
                ### plot conv layer activations            
                whole_act= act[activity]   ## get the whole batch activity                 

                if 'conv' in name:
                    whole_act = np.reshape(whole_act, [len(labels_test), -1, whole_act.shape[2],  whole_act.shape[3]])  ### reshape back to the whole signal shape                
                    sample_layer_act = whole_act[labels_test == label][0:num_sample,...]
                    for sample in range(num_sample):
                        func.plot_conv_activation_with_ori(data_test[labels_test == label][sample, :, 0], sample_layer_act[sample, ...], label, epoch=epoch, save_name=save_name+'/sample-{}-layer_{}-'.format(sample, name))
                ### plot fully connected activations
                elif 'fully' in activity:
                    ipdb.set_trace()
                    whole_act = np.reshape(whole_act, [len(labels_test), -1]).astype(np.float64)  ### reshape back to the whole signal shape
                    plt.figure()
                    plt.imshow(whole_act, interpolation='nearest', aspect='auto')
                    plt.xlabel('# unit in fully connected layer')
                    plt.ylabel('seizure                 unhealthy                  healthy')
                    plt.savefig(save_name+'/EEG data that have maximum activation on #{} unit.eps'.format(No), format='eps')
                    plt.close()
                    for No_unit in range(whole_act.shape[1]):
                        actNo = whole_act[:, No_unit]
                        inds = np.argsort(actNo)[-8:]
                        fig = plt.figure()
                        for i in range(8):
                            ax = fig.add_subplot(4, 2, i+1)
                            plt.plot(data_test[inds[i]], 'royalblue')
                            plt.setp(ax.get_xticklabels(), visible = False)
                            plt.setp(ax.get_yticklabels(), visible = False)
                        plt.title("EEG signals that unit {} is most responsible".format(No_unit))
                        plt.savefig(save_name+'/EEG data that have maximum activation on #{} unit.eps'.format(No), format='eps')
                    func.plotTSNE(whole_act, labels_test, num_classes=num_classes, n_components=2, title="t-SNE", target_names = ['healthy', 'unhealthy', 'seizure'], save_name=save_name+'/fully-activity-epoch{}-sample-{}-'.format(epoch, ), postfix='on Bonn dataset')
                                
                    #sample_layer_act = whole_act[labels_test == label][0:num_sample,...]
                    #for sample in range(num_sample):
                        #func.plot_fully_activation_with_ori(data_test[labels_test == label][sample, :, 0], sample_layer_act[sample,...], label, epoch=epoch, Fs=173.16, NFFT=256, save_name=save_name+'/sample-{}-layer_{}'.format(sample, name))
                        

        #'''TODO: plot 3 samples of each class to interpret the meaning'''
                ##func.put_kernels_on_grid(train_vars[var], pad=1, save_name=save_name+'/conv_kernel', mode='plot')
        #for ind, activation in enumerate(act):
            
            
    #ipdb.set_trace()
    return test_acc, test_loss, logits


def get_batch_data(data, batch_size=128):
    '''load train_data, segment into chuncks of input_dim long
    using generator yield batch
    '''
    while True:
        try:
            for ii in range(len(data)//batch_size):
                batch_y = data[ii*batch_size: (ii+1)*batch_size, 0].astype(np.int)
                batch_x = data[ii*batch_size: (ii+1)*batch_size, 1:]
                
                yield (batch_x, batch_y)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit(1)

            
### construct the network
def train(x):
    
    with tf.name_scope('Data'):
        datas = func.read_data('data/Bonn_data/Bonn_all_shuffle_data.csv', header=0, ifnorm=False)   ##500*4098
        train_data, test_data = train_test_split(datas, test_size=0.2, random_state=19974)
        train_batch = get_batch_data(train_data, batch_size=batch_size)
        num_train = len(train_data)

        print("num_train", num_train, 'sample', train_data[0, 0:5])
    ################# Constructing the network ###########################
    #outputs = mod.fc_net(x, hid_dims=[500, 300, 100], num_classes = num_classes)   ##
    #outputs, kernels = mod.resi_net(x, hid_dims=[1500, 500], seq_len=height, width=width, channels=channels, num_blocks=5, num_classes = num_classes)
    #outputs, kernels = mod.CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], pool_size=[4, 1], strides=[4, 1], seq_len=height, width=width, channels=channels, num_classes = num_classes)
    #outputs, kernels = mod.CNN_new(x, output_channels=[4, 8, 16, 32], num_block=2, num_seg=num_seg, seq_len=height, width=width, channels=channels, num_classes = num_classes)    ## ok
    #outputs, kernels = mod.DeepConvLSTM(x, output_channels=[4, 8, 8], filter_size=[11, 1], pool_size=[4, 1], strides=[2, 1], num_lstm=64, group_size=4, seq_len=height, width=width, channels=channels, num_classes = num_classes)  ## ok
    #outputs, kernels = mod.RNN(x, num_lstm=128, seq_len=height, width=width, channels=channels, group_size=32, num_classes = num_classes)   ##ok
    #outputs, kernels = mod.Dilated_CNN(x, output_channels=16, seq_len=seq_len, width=width, channels=channels, num_classes = num_classes)
    #outputs, kernels = mod.Atrous_CNN(x, output_channels_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [5, 1], seq_len=height, width=width, channels=channels, num_classes = 2)
    #outputs, kernels = mod.PyramidPoolingConv(x, output_channels=[2, 4, 8, 16, 32], filter_size=7, dilation_rate=[2, 8, 16, 32], seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs, kernels = mod.Inception(x, filter_size=[5, 9],num_block=2, seq_len=height, width=width, channels=channels, num_seg=num_seg, num_classes=num_classes)
    #outputs, kernels = mod.Inception_complex(x, output_channels=[4, 8, 16, 32], filter_size=[5, 9], num_block=2, seq_len=height, width=width, channels=channels, num_classes=num_classes)
    #if model_name == 'ResNet': outputs, kernels = mod.ResNet(x, num_layer_per_block=3, filter_size=[[5, 1], [3, 1]], output_channels=[16, 32, 64], pool_size=[[2, 1]], strides=[2, 1], seq_len=height, width=width, channels=channels, num_classes=num_classes)
    #outputs, kernels = mod.AggResNet(x, output_channels=[8, 16, 32], num_stacks=[3, 3, 3], cardinality=8, seq_len=height, width=width, channels=channels, filter_size=[3, 1], pool_size=[2, 1], strides=[2, 1], fc=[100], num_classes=num_classes)

    if model_name == 'CNN_Tutorial': outputs, kernels, activities = mod.CNN_Tutorial(x, output_channels=[16, 16, 16], seq_len=height, width=width, channels=channels, num_classes=num_classes, pool_size=[4, 1], strides=[4, 1], filter_size=[[9, 1], [5, 1]], fc=[200]) ## works on CIFAR, for BB pool_size=[4, 1], strides=[4, 1], filter_size=[9, 1], fc1=200 works well.
   
    #if model_name == 'CNN_Tutorial_Resi': outputs, kernels = mod.CNN_Tutorial_Resi(x, output_channels=[8, 16, 32, 32], seq_len=height, width=width, channels=1, pool_size=[3, 1], strides=[2, 1], filter_size=[[9, 1], [5, 1]], num_classes=num_classes, fc=[200])
    #outputs, kernels = mod.RNN_Tutorial(x, num_rnn=[50, 50], seq_len=height, width=width, channels=channels, fc=[50, 50], group_size=1, drop_rate=0.5, num_classes = num_classes)
    #ipdb.set_trace()
    #### specify logdir
    results_dir= 'results/' + version + '/cpu-batch{}/seq_len{}-slide{}-conv-16-16-16-p4-s4-f5-f3-'.format(batch_size, seq_len, ifslide)+ datetime
    #cnv4_lstm64testcrop10000-add-noise-CNN-dropout0.3-'.format(batch_size, num_seg, majority_vote), seg_len80-conv8-16-32-f9-f5-p3-s2-fc200-lr0.01-, seg_len80-gru50-50-fc50-fc50-drop0.5-
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
        acc_per_class = tf.metrics.mean_per_class_accuracy(tf.argmax(y, 1), predictions, num_classes)
        ### auc always 0.0, https://stackoverflow.com/questions/49887325/tensorflow-always-getting-an-auc-value-of-0
        area_under_curve = tf.contrib.metrics.streaming_auc( predictions=outputs, labels=y, name='auc')[1]

        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('auc', area_under_curve)
        
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                    beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(cost)###,
    #optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)   ### laerning rate 0.01 works
    #optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    #ipdb.set_trace()
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)
    tf.set_random_seed(rand_seed)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        #ipdb.set_trace()
        try:
            saved_global_step = func.load_model(saver, sess, restore_from)
            if is_overwritten_training or saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1

        except:
            print('Something went wrong while restoring checkpoint. '
                  'We will terminate training to avoid accidentally overwriting '
                  'the previous model.')
            raise
            
        
        
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)
        #sess.run(iter.initializer) # every trial restart training
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
        for epoch in range(epochs):
            acc_epoch_train = 0
            loss_epoch_train = 0
            for batch in range(num_train//batch_size + 1):#####                
                save_name = results_dir + '/' + 'step{}_'.format( batch)

                data_train, labels_train = train_batch.next()   ### python2
                #data_train, labels_train = train_batch.__next__()   ### python3
                data_train = np.expand_dims(data_train, 2)

                ## data augmentation
                ### randomly crop a target_len
                if ifcrop:
                    data_train = func.random_crop(data_train, crop_len=crop_len)

                ### add random noise
                #data_train = func.add_random_noise(data_train, prob=0.5, noise_amp=0.01)

                labels_train_hot = np.eye((num_classes))[labels_train.astype(int)] # get one-hot lable

                #if epoch == 0 and batch == 0:
                    #func.plot_BB_training_examples(data_train[0:10, :, :], labels_train[0:10], save_name=save_name)
                #ipdb.set_trace()
                if ifslide:
                    data_slide = func.slide_and_segment(data_train, num_seg=num_seg, window=window, stride=stride )## 5s segment with 1s overlap
                    #ipdb.set_trace()
                    data_train = data_slide
                    labels_train_hot = np.repeat(labels_train_hot, num_seg, axis=0).reshape(-1, labels_train_hot.shape[1])
                #ipdb.set_trace()
                _, summary, acc, c = sess.run([optimizer, summaries, accuracy, cost], feed_dict={x: data_train, y: labels_train_hot, learning_rate:func.lr(epoch)})# , options=options, run_metadata=run_metadata We collect profiling infos for each step.
                writer.add_summary(summary, epoch*(num_train//batch_size)+batch)##
                
                ## accumulate the acc and cost later to average
                acc_epoch_train += acc
                loss_epoch_train += c
                if batch % 50 == 0:
                    print('epoch', epoch, 'batch:',batch, 'loss:', c, 'train-accuracy:', acc)
            ###################### test ######################################
            if epoch % 1 == 0:
                acc_epoch_test, loss_epoch_test, logits = evaluate_on_test(sess, epoch, accuracy, cost, outputs, test_data, kernels, crop_len=crop_len, ifslide=ifslide, activities=activities, ifnorm=ifnorm, ifcrop=ifcrop, header=header, save_name=results_dir)
                print('epoch', epoch, 'batch:',batch, 'loss:', c, 'train-accuracy:', acc, 'test-accuracy:', acc_epoch_test)
            ######################################################## activities,
                
            # track training and testing
            loss_total_train.append(loss_epoch_train / (batch + 1))            
            acc_total_train.append(acc_epoch_train / (batch + 1))
            loss_total_test.append(loss_epoch_test)            
            acc_total_test.append(acc_epoch_test)
            
            if epoch % func.get_save_every(epoch) == 0:
                func.save_model(saver, sess, logdir, epoch)
                last_saved_step = epoch

            #if epoch == 1:
                #variables = sess.run(kernels, feed_dict={x: data_train, y: labels_train_hot, learning_rate:func.func.lr(epoch)})
            
            if epoch % 5 == 0:
                
                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], ifsmooth=False, hlines=[0.8, 0.85, 0.9], window_len=smooth_win_len, xlabel= 'training epochs', ylabel='accuracy', colors=['darkcyan', 'm'], ylim=[0.0, 1.05], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ '/learning_curve_epoch_{}_seed{}'.format(epoch, rand_seed))

                func.plot_smooth_shadow_curve([loss_total_train, loss_total_test], window_len=smooth_win_len, ifsmooth=False, hlines=[], colors=['c', 'violet'], ylim=[0.05, 1.5], xlabel= 'training epochs', ylabel='loss', title='Loss',labels=['training loss', 'test loss'], save_name=results_dir+ '/loss_epoch_{}_seed{}'.format(epoch, rand_seed))

                func.save_data_to_csv((acc_total_train, loss_total_train, acc_total_test, loss_total_test), header='accuracy_train,loss_train,accuracy_test,loss_test', save_name=results_dir + '/' + datetime + 'batch_accuracy_per_class.csv')   ### the header names should be without space! TODO
                #func.plot_auc_curve(labels_train_hot, outputs, save_name=results_dir+'/epoch_{}'.format(epoch))
    ##Stop the threads
    #coord.request_stop()
    
    ##Wait for threads to stop
    #coord.join(threads)


if __name__ == '__main__':
    train(x)
### define the cost and opti
