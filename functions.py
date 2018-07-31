#### util functions for EEG classification
##### file operations
##### data manipulation
##### fploting
import fnmatch
import numpy as np
import tensorflow as tf
import csv
import codecs
from scipy.signal import decimate    ###downsampling
import multiprocessing
import os
import sys
from functools import partial      ### for multiprocessing
import matplotlib.pyplot as plt
import ipdb
import random
from scipy.stats import zscore
import pandas as pd

# import scipy.stats as stats
import matplotlib.pylab as pylab
params = {'legend.fontsize': 12,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
import matplotlib

########################### model ######################
def lr(epoch):
    learning_rate = 0.001
    if epoch > 120:
        learning_rate *= 0.5e-3
    elif epoch > 100:
        learning_rate *= 1e-3
    elif epoch > 50:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate

def get_save_every(epoch):
    save_every = 2
    if epoch > 30:
        save_every = 10
    elif epoch > 9:
        save_every = 5

    return save_every
    
def save_model(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')
    

def load_model(saver, sess, save_dir):
    #print('Trying to restore saved checkpoints from {} ...'.format(logdir),
          #end='')
    ckpt = tf.train.get_checkpoint_state(save_dir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print('  Global step was: {}'.format(global_step))
        print('  Restoring...')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(' Done.')
        return global_step
    else:
        print(' No checkpoint found.')
        return None
        

###################### files operation##########################
def find_files(directory, pattern='Data*.csv', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'Data' in pattern:
                    if 'Data_F' in filename:
                        label = '1'
                    elif 'Data_N' in filename:
                        label = '0'
                elif 'segNorm' in filename:
                    if 'baseline' in filename:
                        label = '0'
                    elif 'tip-off' in filename:
                        label = '1'
                    elif 'seizure' in filename:
                        label = '2'
                else:
                    if 'Z' or 'O' in filename:
                        label = '0'
                    elif 'N' or 'F' in filename:
                        label = '1'
                    elif 'S' in filename:
                        label = '2'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    print(len(files))
    random.shuffle(files)
    return files 


def rename_files(filename):
    #os.rename(filename, os.path.dirname(filename) + '/Data_' + os.path.basename(filename)[5:])
    os.rename(filename, os.path.dirname(filename)+'/Bonn_'+os.path.basename(filename))

def remove_files(filename):
    os.remove(filename)

def multiprocessing_func(data_dir):
    '''PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed
    PLEASE disable the import ipdb!!!'''
    filenames = find_files(data_dir, pattern='*.csv', withlabel=False )
    print(filenames)
    #ipdb.set_trace()
    pool = multiprocessing.Pool()
    version =  'rename'  #"augment_data" #'remove'       #  'rename'  #'downsampling' #'save_tfrecord'#     #None#'rename'      # 'rename'        #
    if version == 'downsampling':
        # for ds in [2]:
        pool.map(partial(downsampling, ds_factor=2), filenames)
        print("Downsampling Done!")
    elif version == 'rename':
        pool.map(rename_files, filenames)
        print("rename Done!")
    elif version == 'remove':
        pool.map(remove_files, filenames)
        print("remove Done!")
    elif version == "save_tfrecord":
        pool.map(read_data_save_tfrecord, filenames)
        print("tfrecord saved")
    elif version == "augment_data":
        pool.map(augment_data_with_ar1, filenames)
    pool.close()

###################### Data munipulation##########################

def read_data(filename, header=None, ifnorm=True, start=0, width=2 ):
    '''read data from .csv
    Param:
        filename: string e.g. 'data/Data_F_Ind0001.csv'
        ifnorm: boolean, 1 zscore normalization
        start: with filter augmented data, start index of the column indicate which group of data to use
        width: how many columns of data to use, times of 2
    return:
        data: 2d array [seq_len, channel]'''

    data = pd.read_csv(filename, header=header)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data


def get_tfrecords_next_batch(data_dir, pattern='*.tfrecords', seq_len=10240, width=2, channels=1, epochs=50, batch_size=20, withlabel=False):

    '''Get presaved tfrecords files and enqueue into batches
    Param:
        data_dir: data directory

    Return:
        data: batch_size*seq*width*channel
        label: int label'''
        
    files = find_files(data_dir, pattern=pattern, withlabel=withlabel)

    feature = {'data': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    
    ### Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(files, num_epochs=epochs)   ### the files have to a list

    ###Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    #### Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    ####Convert the image data from string back to the numbers
    data = tf.decode_raw(features['data'], tf.float64)
    
    ### Cast label data into int32
    labels = tf.cast(features['label'], tf.int32)   ### the feature name should be exactly the same as you save them
    
    ## define the shape
    data = tf.reshape(data, [seq_len, width, channels])
    ### Creates batches by randomly shuffling tensors
    batch_data, batch_labels = tf.train.shuffle_batch([data, labels], batch_size=batch_size, capacity=50000, num_threads=32, min_after_dequeue=10000)

    return batch_data, batch_labels



    
def augment_data_with_ar1(filename):
    '''Read data from file and compute lag1 autocorrelation and then save (original data, ar1)
    fix value pad the first window size autocorrelation'''
    #ipdb.set_trace()
    print(filename)
    assert 'Data' in filename   ### function for aug the bern-barcelona dataset
    # data = pd.read_csv(filename, names=['1', '2'])
    save_name = os.path.dirname(filename) + '/aug2_' + os.path.basename(filename)[0:-4] + '_aug2.csv'
    aug_data = np.zeros((10240, 4))
    data = read_data(filename, ifnorm=True)
    autocorr_x = lag_ar(data[:, 0], window=1024)
    autocorr_y = lag_ar(data[:, 1], window=1024)
    aug_data[:, 0:2] = data
    aug_data[:, 2] = autocorr_x
    aug_data[:, 3] = autocorr_y
    np.savetxt(save_name, aug_data, header="datax,datay,corrx,corry", delimiter=',', comments='')
    # aug_data.to_csv(os.path.dirname(filename) + '/aug2_' + os.path.basename(filename)[0:-4] + '_aug2.csv', index=False)




def read_data_save_one_csv(data_dir):
    '''find all files and read each file into one line and save all files' data 'in one  csv.
    Each row is one training sample
    return:
        the first element on each row is the label, then followed by 1*20480 data'''
    #ipdb.set_trace()
    filenames = find_files(data_dir, pattern='*_aug.csv', withlabel=False)
    whole_csv = 'data/test_data.csv'
    #whole_csv = 'data/test_files/test_files.csv'
    whole_data = []
    for ind, filename in enumerate(filenames):
        if 'Data_F' in filename:
            label = 1
        elif 'Data_N' in filename:
            label = 0
        if ind%19 == 0:
            print("ind", ind, "out of ", len(filenames))
        data = read_data(filename, ifnorm=True )   ### falttened data 1 * 20480
        #ipdb.set_trace()
        data = np.hstack((label, data))  #### [label, data1 * 20480]
        whole_data.append(data)
    np.savetxt(whole_csv, np.array(whole_data), header='label, flattened data', delimiter=',', fmt="%10.5f", comments='')

def load_and_save_data_to_npz(data_dir, pattern='Data*.csv', withlabel=True, ifnorm=True, num_classes=2, save_name='data', seq_len=10240, width=2):
    '''Keras way of loading data
    return: x, y, x_test, y_test
        x: [num_samples, seq_len, channel]
        y: [num_samples, ]  ## int label
        x: [num_samples, seq_len, channel]
        x: [num_samples, ]  ## int label
        '''
    #### Get data
    files_wlabel = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the 
    files, labels = np.array(files_wlabel)[:, 0], np.array(np.array(files_wlabel)[:, 1]).astype(np.int)

    datas = np.zeros([len(files), seq_len, width])
    for ind in range(len(files)):
        if ind % 100 == 0:
            print('files', files[ind], 'label', labels[ind])
        data = read_data(files[ind], ifnorm=ifnorm)
        datas[ind, :, :] = data

    np.savez(data_dir + "/" + save_name, data=datas, label=np.array(labels))
    
    return datas, np.array(labels)


def downsampling(filename, ds_factor):
    """Downsample the original data by the factor of ds_factor to get a low resolution version"""
    x, y = read_data(filename)
    ds_x =  decimate(x, ds_factor)
    ds_y =  decimate(y, ds_factor)
    np.savetxt(os.path.dirname(filename) + "/ds_" + np.str(ds_factor)  + os.path.basename(filename) , zip(ds_x, ds_y), delimiter=',', fmt="%10.5f")

def save_data_to_csv(data, header='data', save_name="save_data"):
    '''save data into a .csv file
    data: list of data that need to be saved, (x1, x2, x3...)
    header: String that will be written at the beginning of the file.'''
    np.savetxt(save_name, data, header=header, delimiter=',', fmt="%10.5f", comments='')

def smooth(x, window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    """
    #if x.ndim != 1:
        #raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def opp_slide2segment(data_x, data_y, ws, ss):
    '''apply a sliding window to original data and get segments of a certain window lenght
    e.g.
    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_slide2segment(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    '''
    data_x = slide2segment(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in slide2segment(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def slide_and_segment(data_x, num_seg=5, window=128, stride=64):
    '''
    Param:
        datax: array-like data shape (batch_size, seq_len, channel)
        data_y: shape (num_seq, num_classes)
        num_seg: number of segments you want from one seqeunce
        window: int, number of frames to stack together to predict future
        noverlap: int, how many frames overlap with last window
    Return:
        expand_x : shape(batch_size*num_segment, window, channel)
        expand_y : shape(num_seq*num_segment, num_classes)
        '''
    assert len(data_x.shape) == 3
    #if data_x.shape[1] % num_seg == 0:   ## if it's int segment
        #num_seg = (data_x.shape[1] - np.int(window)) // stride + 1
    #else:
        #num_seg = (data_x.shape[1] - np.int(window)) // stride
        
    expand_data = np.zeros((data_x.shape[0], num_seg, window, data_x.shape[-1]))
    #ipdb.set_trace()
    for ii in range(data_x.shape[0]):
        
        shape = (num_seg, window, data_x.shape[-1])      ## done change the num_seq
        strides = (data_x.itemsize*stride*data_x.shape[-1], data_x.itemsize*data_x.shape[-1], data_x.itemsize)
        expand_x = np.lib.stride_tricks.as_strided(data_x[ii, :, :], shape=shape, strides=strides)
        expand_data[ii, :, :, ] = expand_x
    #ipdb.set_trace()
    #expand_y = np.repeat(data_y,  num_seg, axis=0).reshape(-1, data_y.shape[1])
    return expand_data.reshape(-1, window, data_x.shape[-1])#, expand_y


def lag_ar(data, window=1024, lag=1) :
    """
    data:  1D array, the whole data    https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781783553358/7/ch07lvl1sec75/autocorrelation
    return:
    the alg1 correlation coefficient given the lag and window size"""
    data = np.array(data)
    lag_ar = np.zeros((data.size))
    for ii in range(data.size-window):  ###1000-200
        ar1 = np.corrcoef(np.array([data[ii:window+ii], data[ii+lag:window+ii+lag]]))
        if ii == 0:
            lag_ar[0:window+1] = ar1[0, 1]
        else:
            lag_ar[window+ii] = ar1[0, 1]
    return lag_ar

def filter_loss(data, threshold = 20):
    ''' get all the data loss(value doesn't change from previous time. ')
    param:
    problem: there are a lot places where only 1/2 data is repeated. they could be good just discard those repeated values
        '''
    error = data[0:-1] - data[1:]
    indices = np.where(error==0)        ### get the indices of the start of data loss
    record_intervals = indice[1:] - indice[0:-1]   ### get record data points interval between two consecutive data loss points. if it's > threshold, then it can be used as good data recording
    start_meaning_record = np.where(record_intervals >threshold)   ### pionts back to between which
              #####data loss there is good recording. from data[start_meaning_record] lasts for record_intervals
    start_meaning_record = start_meaning_record[0]
    ### get the starting point of a good recording segment and duration of it
    ### start index of a long/good recording
    start = indice[start_meaning_record][0]+1
    duration = record_intervals[start_meaning_record[0]]   ### the duration
    data_seg = data[start : start+ duration]    ### finally get the data
    data_segs.append(data_seg)
    return data_segs

def split_filter_data_with_long_loss(data, accept_loss_threshold=50, accept_data_len=2048):
    '''#### start new filter.
    ### 1. discard repeated data if the number of repeatation is below a threshold(10).
        ### it shouldn't make a hug difference since the sampling reate is 512Hz
    ### 2. then segement the recording with long data loss (if the data loss over a threshold, split the data into segments)
    ### 3. discard short recordings
    ### 4. leftover segments with no data repeatation and long enough
    Param:
        data: 1D array with data losses
        accept_loss_threshold: int, below the threshold, the data can be interpolated
        acccep_data_len: if after segmentation, the data is shorter than this, will be discarded
    return:
        data_segs: shape=[num_seg, variable(seq_len), 1]'''
    error = data[0:-1] - data[1:]
    non_zero_error_ind = np.where(error!=0)[0]   ### those indices where there is no data loss

    loss_intervals = non_zero_error_ind[1:] - non_zero_error_ind[0:-1]

    # ## if smaller than threshold, it means the data loss is short and make sense to squize out the loss points
    long_loss_interval_start = np.where(loss_intervals>accept_loss_threshold)[0]   ##get where there are long(>threshold) data loss

    accepted_segs_ind = np.split( non_zero_error_ind, long_loss_interval_start+1)

    data_segs = []
    for ii, ind_seg in enumerate(accepted_segs_ind):
        if ind_seg.size > accept_data_len:
            ### interpolate the missing data
            data[ind_seg].shape
            data_seg_interp = linear_interpolation(data[ind_seg])
            data_seg = data_seg_interp  ### get segments between long data loss
            print("data_seg_interp.shape", data_seg_interp)
            data_segs.append(data_seg[0:data_seg.size-data_seg.size%accept_data_len])  ## discard the data at the end

    return np.array(data_segs)


def linear_interpolation(data):
    """Helper to handle indices and logical indices of repeated data points(data loss points).

    Input:
        - data, 1d numpy array with possible data loss which appears with repeating previous values
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices

    """
    err = np.insert((data[1:] - data[0:-1]), 0, data[0])   ## the err of the first element is itself
    zeros_ind = err == 0
    x = lambda z:z.nonzero()[0]

    data[zeros_ind] = np.interp(x(zeros_ind) ,x(~zeros_ind), data[~zeros_ind])

    return data

def add_random_noise(data, prob=0.5, noise_amp=0.02):
    '''randomly add noise to original data
    param:
        data: 2D array: batch_size*seq_len*width
        '''
    shape = data.shape
    mask = np.random.uniform(0, 1, shape)
    mask[mask > prob] = 1
    mask[mask <= prob] = 0

    noise = noise_amp * np.random.randn(data.size).reshape(shape)
    noise = noise * mask
    data = data + noise

    return data


def random_crop(data, crop_len=10000):
    '''given a target seq len, randomly crop the data'''
    choice = data.shape[1] - crop_len
    start = np.random.choice(choice, data.shape[0])
    crop_data = np.array([data[i, start[i]:start[i]+crop_len] for i in range(data.shape[0])])

    return crop_data


###################### plots ##########################
def plot_learning_curve(train_scores, test_scores , num_trial=1, title="Learning curve", save_name="learning curve"):
    '''plot smooth learning curve
    train_scores: n_samples * n_features
    test_scores: n_samples * n_features
    '''
    plt.figure()
    plt.title(title)
    plt.xlabel("training batches")
    plt.ylabel("accuracy")
    if num_trial > 1:
        train_sizes = len(train_scores)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(np.arange(train_sizes), train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.25,color="m")
        plt.fill_between(np.arange(train_sizes), test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.25, color="c")
        plt.plot(np.arange(train_sizes), train_scores_mean, '-', color="indigo",
                 label="Training score")
        plt.plot(np.arange(train_sizes), test_scores_mean, '-', color="teal",
                 label="Test score")
        plt.legend(loc="best")
    elif num_trial == 1:
        train_scores_mean = smooth(train_scores_mean, window=11)
        data_smooth = train_scores_mean.shape[0]
        sizes = data.shape[0]
        plt.grid()
        plt.fill_between(np.arange(train_scores), data_smooth - data_std, data_smooth + data_std, alpha=0.3, color="lightskyblue")
        plt.plot(np.arange(sizes), data_smooth, '-', color="royalblue")
    plt.savefig(save_name, format="pdf")
    plt.close()

def plot_smooth_shadow_curve(datas, ifsmooth=False, hlines=[0.8, 0.85], ylim=[0, 1.05], window_len=25, colors=['darkcyan'], xlabel='training batches / 20', ylabel='accuracy', title='Loss during training', labels='accuracy_train', save_name="loss"):
    '''plot a smooth version of noisy data with mean and std as shadow
    data: a list of variables values, shape: (batches, )
    color: list of prefered colors
    '''
    plt.figure()
    fill_colors = ['lightcoral', 'plum']
    if ifsmooth:
        for ind, data in enumerate(datas) :
            data_smooth = smooth(data, window_len=window_len)
            data_smooth = data_smooth[0:len(data)]
            data_mean = np.mean(np.vstack((data, data_smooth)), axis=0)
            data_std = np.std(np.vstack((data, data_smooth)), axis=0)
            sizes = data_std.shape[0]
            plt.grid()
            plt.fill_between(np.arange(sizes), data_smooth - data_std, data_smooth + data_std, alpha=0.5, color=fill_colors[ind])
            plt.plot(np.arange(sizes), data_smooth, '-', linewidth=2, color=colors[ind], label=labels[ind])
        for hline in hlines:
            plt.hlines(hline, 0, np.arange(sizes), linestyle='--', colors='salmon',  linewidth=1.5)
    else:
        for ind, data in enumerate(datas) :
            plt.plot(data, '*-', linewidth=2, color=colors[ind], label=labels[ind])
        for hline in hlines:
            plt.hlines(hline, 0, np.array(data).size, linestyle='--', colors='salmon',  linewidth=1.5)
            
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(ylim)
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(save_name+'.png', format="png")
    plt.close()


def plot_test_samples(samples, true_labels, pred_labels, save_name='results/'):
    plt.figure()
    for ii in range(20):
        ax1 = plt.subplot(5, 4, ii +1)
        plt.plot(np.arange(samples[ii, :, 0])/ 512.0, samples[ii, :, 0])
        plt.xlim([0, samples[ii, :, 0]/ 512.0])
        plt.xlabel("{}-{:10.4f}".format(true_labels[ii], np.max(pred_labels[ii, :])))
        #plt.setp(ax1.get_yticklabels(), visible = False)
        plt.title("True - Predict")
        plt.setp(ax1.get_xticklabels(), visible = False)
    plt.tight_layout()
    plt.savefig(save_name + 'samples_test.pdf', format = 'pdf')
    plt.close()

       
def visualize_fc_layer_activation(sess, layer_name, inputs, save_name='results/'):
    '''visualize fc layer activation given some inputs
    param: 
        sess: current session
        layer_name: the layer you want to visualize
        inputs: 2D array [batch_size, seq_len, width]
        
    return:
        activations: activations from each layer'''

    ## get all the viariables with the layername
    with tf.variable_scope(layer_name, reuse=True):
        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
    
    weights = []
    biases = []
    kernels = []
    ### get all weights and biases
    for ind, var_name in enumerate(vars_list):
        if 'weighs' in var_name:
            weights.append(var_name)
        elif 'bias' in var_name:
            biases.append(var_name)
        elif 'kernel' in var_name:
            kernels.append(var_name)
##  batch*1024*2 -- batch*2048*1 --> w 2048*500 --> w 500*300 --> w300*2  -->batch*2
    for ii in range(len(weights)):

        w = tf.cast(weights[ii], tf.float64)
        b = tf.cast(biases[ii], tf.float64)
    
        example = tf.placeholder(tf.float32, [None, inputs.shape[1], inputs.shape[2]])
        acti_mat = tf.nn.relu(tf.matmul(tf.cast(tf.transpose(example), tf.float64), weights) + bias)
        activation = sess.run(acti_mat,  feed_dict={example: inputs})
        plt.imshow(acti_matrix, interpolation="nearest", cmap="gray", aspect="auto")

        plt.savefig(save_name + "fc_activation.png", format="png")
        plt.close()
        inputs = activation
    
    acti_mat_all = []
    acti_mat = np.zeros((inputs.shape[0], bias.shape[0]))



def plot_train_samples(samples, true_labels, xlabel='label: 0', ylabel='value', save_name='results/'):
    plt.figure()
    for ii in range(20):
        ax1 = plt.subplot(5, 4, ii +1)
        #plt.plot(np.arange(samples[ii, :, 0])/ 512.0, samples[ii, :, 0])
        plt.imshow(samples[ii], interpolation='nearest', aspect='auto')
        #plt.xlim([0, samples[ii]/ 512.0])
        #plt.xlim([0, samples[ii]])
        plt.xlabel("label: " + np.str(true_labels[ii]))
        if not ylabel:
            plt.ylabel(ylabel)
        #plt.setp(ax1.get_yticklabels(), visible = False)
        plt.setp(ax1.get_xticklabels(), visible = False)
        plt.setp(ax1.get_yticklabels(), visible = False)
    plt.tight_layout()
    plt.savefig(save_name + 'samples_train.png', format = 'png')
    plt.close()

def plot_BB_training_examples(samples, true_labels, save_name='results/'):
    
    for ii in range(3):
        plt.figure()
        for ind in range(samples[ii, :, :].shape[-1]):
        
            ax1 = plt.subplot(samples[ii, :, :].shape[-1], 1, ind+1)
            plt.plot(np.arange(samples[ii, :, ind].size)/ 512.0, samples[ii, :, ind], label="data_{}".format(ind+1))
            plt.ylabel("amplitude ")
            plt.xlabel("time / s")
            plt.legend()
            plt.xlim([0, samples[ii, :, 0].size/ 512.0])
        plt.xlabel("data samples, label={}".format(true_labels[ii]))
        plt.tight_layout()
        plt.savefig(save_name + "vis_train_data{}.png".format(ii), format="png")
        plt.close()


def plotTSNE(data, labels, num_classes=2, n_components=3, title="t-SNE", target_names = ['non_focal', 'focal'], save_name='/results', postfix='band_PSD'):
    '''tsne clustering on data
    param:
        data: 2d array shape: batch*seq_len*width
        label: 1d array, int labels'''

    from tsne import bh_sne
    tsne_results = bh_sne(data, d=n_components)
    #tsne_results = TSNE(n_components=3, random_state=99).fit_transform(data)
    
    #colors =plt.cm.get_cmap("cool", num_classes)
    cmap=['orchid', 'fuchsia',  'indigo', 'aqua', 'darkturquoise', 'mediumseagreen', 'darkgreen','slateblue', 'royalblue', 'cornflowerblue', 'navy',  'mediumaquamarine', 'lightcoral']
    colors = np.random.choice(cmap, num_classes)

    fig = plt.figure()
    if n_components == 3:
        vis_x = tsne_results[:, 0]
        vis_y = tsne_results[:, 1]
        vis_z = tsne_results[:, 2]
        ax = fig.add_subplot(111, projection='3d')        
        for i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], tsne_results[labels == i, 2], color=color, alpha=.8,label=target_name)
    elif n_components == 2:
        vis_x = tsne_results[:, 0]
        vis_y = tsne_results[:, 1]
        ax = fig.add_subplot(111)
        for color, i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], color=color, alpha=.8, label=target_name)###lw=2,
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("cool", num_classes))   ##
    plt.title("t-SNE-{}".format(postfix))
    plt.savefig(save_name+"t-SNE-{}.png".format(postfix), format='png')
    plt.close()


def plot_PCA(data, labels, n_components=3, num_classes=2, colors = ['navy', 'turquoise'], target_names = ['non-focal', 'focal'], title='PCA', postfix='band_PSD'):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca_results = pca.fit(data).transform(data)
    cmap=['orchid', 'fuchsia',  'indigo', 'aqua', 'darkturquoise', 'mediumseagreen', 'darkgreen','slateblue', 'royalblue', 'cornflowerblue', 'navy',  'mediumaquamarine', 'lightcoral']
    colors = np.random.choice(cmap, num_classes)
    lw = 2
    fig = plt.figure()
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        for color, i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(pca_results[labels == i, 0], pca_results[labels == i, 1], pca_results[labels == i, 2], color=color, alpha=.8, lw=lw, label=target_name)
    elif n_components == 2:
        ax = fig.add_subplot(111)
        for color, i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(pca_results[labels == i, 0], pca_results[labels == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("PCA on {}".format(postfix))
    plt.savefig(save_name+"PCA-on {}.png".format(postfix), format='png')
    plt.close()
#def plotOnePair(data):
    #'''plot the original data-pair'''
def plot_bar_chart():
    plt.figure()
    barWidth = 0.5
    r1 = np.arange(6)+0.25
    #r2 = [x + barWidth for x in r1]
    plt.bar(r1, mean_new, color='deepskyblue', width=barWidth, edgecolor='white', label='test_accuracy')
    plt.ylim([0.5, 1.0])
    plt.hlines(xmin=0, xmax=6, y=0.8, linestyle='--', color='plum', linewidth=2)
    plt.hlines(xmin=0, xmax=6, y=0.9, linestyle='--', color='plum', linewidth=2)

    #plt.bar(r2, mean_train, color='m', width=barWidth, edgecolor='white', label='train_accuracy')
    plt.legend()
    plt.title("Classification accuracy comparison")
    plt.xticks([r + barWidth for r in range(len(mean_test))], keys_new, rotation=0)

    for ind in range(len(mean_test)):
        
        plt.text(r1[ind]+ 0.25*barWidth, mean_new[ind]+0.005, '{0:.3f}'.format(mean_new[ind]), size = 18)

def plot_auc_curve(labels, predictions, save_name='results/'):
    '''plot the auc curve'''
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    print("sklearn auc: ", roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC (curve area = %0.2f)'% roc_auc)
    plt.plot([0, 1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='best')
    plt.savefig(save_name + 'auc_curve.png', format='png')
    plt.close()

def put_kernels_on_grid(kernel, pad = 1, save_name='kernel', mode='imshow'):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    https://gist.github.com/kukuruza/03731dc494603ceab0c5
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
                
    (grid_Y, grid_X) = factorization(kernel.shape[3])

    print ('grid: %d = (%d, %d)' % (kernel.shape[3], grid_Y, grid_X))

    if mode == 'plot':
        pad = 0
    
    x_min = np.min(kernel)
    x_max = np.max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)  ### normalize the kernel

    # pad X and Y
    x = np.pad(kernel,((pad,pad),(pad, pad),(0,0),(0,0)), 'constant', constant_values=0)
    # X and Y dimensions, w.r.t. padding
    Y = kernel.shape[0] + 2 * pad
    X = kernel.shape[1] + 2 * pad
    #ipdb.set_trace()
    channels = kernel.shape[2]    ## in channels
    #print('channels', channels)
    # put NumKernels to the 1st dimension
    x = np.transpose(x, (3, 0, 1, 2)) ###(16, 7, 3, 8)
    #print('x.shape', x.shape)
    # organize grid on Y axis
    x = x.reshape(grid_X, Y * grid_Y, X, channels)   ###(4, 28, 3, 8)
    #print('x.shape', x.shape)
    # switch X and Y axes
    x = np.transpose(x, (0, 2, 1, 3))       ##(4, 3, 28, 8)
    #print('x.shape', x.shape)
    # organize grid on X axis
    x = np.reshape(x, [1, X * grid_X, Y * grid_Y, channels])  ###(1, 12, 28, 8)
    #print('x.shape', x.shape)
    # back to normal order (not combining with the next step for clarity)
    x = np.transpose(x, (2, 1, 3, 0))   ###(28, 12, 8, 1)
    #print('x.shape', x.shape)
    # to np.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = np.transpose(x, (3, 0, 1, 2))   ###(1, 28, 12, 8)
    #print('x.shape', x.shape)

    plt.figure()
    ipdb.set_trace()
    for ind in range(x.shape[-1]):
        if mode == 'imshow':
            plt.imshow(x[0, :, :, ind], interpolation='nearest', aspect='auto')
            plt.savefig(save_name + "-channel{}.png".format(ind), format='png')
            plt.close()
        if mode == 'plot':

            plt.plot(x[0, :, ind, 0])
            plt.savefig(save_name + "-channel{}.png".format(ind), format='png')
            plt.close()


def add_conved_image_to_summary(net, save_name='results/'):
    '''given the output of a con2d layer, visualize the convolution results
    Param:
        net: shape (batch_size, height, width, channels)'''
    
    def factorization(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
    #ipdb.set_trace()
    channels = net.shape[3].value
    ix, iy = net.shape[2].value, net.shape[1].value
    print("ix, iy", ix, iy)
    (cy, cx) = factorization(net.shape[3].value)
    pad = 2
    #ipdb.set_trace()
    for ind in range(3):
        image_ori = net[ind,...]
        print("image_ori", image_ori.shape)
        #ipdb.set_trace()
        image = tf.pad(image_ori, tf.constant( [[pad,pad],[pad, pad],[0,0]] ), mode = 'CONSTANT')
        ix_pad = image.shape[1].value
        iy_pad = image.shape[0].value
        image = tf.image.resize_image_with_crop_or_pad(image, iy_pad, ix_pad)
        print("image_ori", image.shape)
        image = tf.reshape(image,(iy_pad,ix_pad,cy,cx))
        print("image_ori", image.shape)
        image = tf.transpose(image,(2,0,3,1)) #cy,iy,cx,ix

        #net = np.pad(net,((pad,pad),(pad, pad),(0,0)), 'constant', constant_values=0)
        #net = np.reshape(net,(iy,ix,cy,cx))
        #net = np.transpose(net,(2,0,3,1))

        ### image_summary needs 4d input
        image= tf.reshape(image,(1,cy*iy_pad,cx*ix_pad,1))
        tf.summary.image('sample_conved_input_{}'.format(ind), image)
        #net= np.reshape(net,(1,cy*iy,cx*ix,1))
        #plt.imshow(net[0, :, :, 0], interpolation='nearest', aspect='auto')
        #plt.savefig(save_name+'-net_output.png', format='png')
        #plt.close()

def plot_fully_activation_with_ori(original_data, activation, label, epoch=0, Fs=173.16, NFFT=256, save_name='results/'):
    '''plot the fully connected layer activation with original signal and its spectrogram
    Param:
        original_data: 1D array seq_len*1
        activation: 1d array 1*(num_seg*fully_unit))   # concat the segments from one sample together
        label: int'''

    fig = plt.figure(figsize=(10, 8))
    ori_len = original_data.size
    ### plot the original signal for visualizaiont
    ax0 = fig.add_subplot(3, 1, 1)
    plt.plot(np.arange(ori_len)/Fs, original_data, 'm')
    plt.xlim([0, ori_len/Fs])
    plt.title('Original signal (lable: {})'.format(label))
    plt.xlabel('time / s')
    ### plot spectrogram
    ax1 = fig.add_subplot(3, 1, 2)
    spec = plt.specgram(original_data, NFFT=NFFT, Fs=Fs)
    (Spec, f, t) = spec[0], spec[1], spec[2]             
    plt.title('Spectrogram of orginal signal')
    plt.xlabel('time / s')
    plt.ylabel('frequency')
    plt.xlim([0, t[-1]])
    plt.ylim([0, f[-1]])

    ### plot activation
    ax2 = fig.add_subplot(3, 1, 3)
    plt.plot(activation, 'royalblue')
    plt.ylabel('activation')
    plt.xlim([0, activation.size])
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.setp(ax2.get_yticklabels(), visible = False)
    plt.tight_layout()
    plt.savefig(save_name + '_label_{}_activity-epoch-{}.png'.format(label, epoch), format='png')
    plt.close()

def factorization(n):
    for i in range(int(np.sqrt(float(n))), 0, -1):
        if n % i == 0:
            if i == 1:
                print('Who would enter a prime number of filters')
            return (i, int(n / i))



def plot_conv_activation_with_ori(original_data, activation, label, epoch=0, Fs=173.16, NFFT=256, save_name='results/'):
    '''plot the activiation of all the kernels in a grid with original signal for vis
    Param:
        original_data: 1D array seq_len*1
        activation: height*width*channels
        label: int
        '''
    fig = plt.figure(figsize=(10, 8))
    count = 0
    length = original_data.size
    (grid_Y, grid_X) = factorization(activation.shape[2])        ## get the grid size
    ### plot the original signal for visualizaiont
    ax = fig.add_subplot(grid_Y+1, grid_X, (1, grid_X))
    ax.plot(np.arange(length)/Fs, original_data, 'm')
    plt.xlim([0, length/Fs])
    plt.title('Original signal (lable: {})'.format(label))
    plt.setp(ax.get_xticklabels(), visible = False)
    plt.setp(ax.get_yticklabels(), visible = False)
    plt.xlabel('time / s')
    #### plot the acitivations
    for row in range(grid_Y):                            
        for col in range(grid_X):
            ax = plt.subplot(grid_Y+1, grid_X, count+grid_X+1)
            plt.plot(activation[:, 0, count])                               
            plt.setp(ax.get_xticklabels(), visible = False)
            plt.setp(ax.get_yticklabels(), visible = False)
            plt.xlim([0, activation[:, 0, count].size])
            count += 1
    plt.tight_layout()
    plt.savefig(save_name + '_label_{}_activity-epoch-{}.png'.format(label, epoch), format='png')
    plt.close()
####################### Data munipulation##########################

# #
#if __name__ == "__main__":
 ##     #data_dir = "data/train_data"
 ##     # data_dir_test = "data/test_data"
 ##     #data_dir = 'data/test_files'
 ##     # # read_data_save_tfrecord(data_dir)
    ## ddd = ["data/train_data", "data/test_data"]
     ## ipdb.set_trace()
     ## augment_data_with_ar1(filename)
    ## for direc in ddd:
    ##multiprocessing_func("data/train_data")
 ##     # get_Data(data_dir, data_dir_test, pattern='Data*.csv', withlabel=True)
 ##     # multiprocessing_func(data_dir)
 ##     # read_from_tfrecord("data/test_files/test_files.tfrecords")
 ##     #read_tfrecord()
 ##     #read_data_save_one_csv(data_dir_test)
 ##     #filename = "data/train_data/train_data.csv"
 ##     #read_data(filename)
    #data_dir = "data/Bonn_data/"
    #multiprocessing_func(data_dir)
    ##load_and_save_data(data_dir, pattern='Data*.csv', withlabel=True, num_classes=2)


#def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    #'''Visualize conv. features as an image (mostly for the 1st layer).
    #Place kernel into a grid, with some paddings between adjacent filters.

    #Args:
      #kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      #(grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           #User is responsible of how to break into two multiples.
      #pad:               number of black pixels around each filter (between them)

    #Return:
      #Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    #'''

    #x_min = tf.reduce_min(kernel)
    #x_max = tf.reduce_max(kernel)

    #kernel1 = (kernel - x_min) / (x_max - x_min)

    ## pad X and Y
    #x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    ## X and Y dimensions, w.r.t. padding
    #Y = kernel1.get_shape()[0] + 2 * pad
    #X = kernel1.get_shape()[1] + 2 * pad

    #channels = kernel1.get_shape()[2]

    ## put NumKernels to the 1st dimension
    #x2 = tf.transpose(x1, (3, 0, 1, 2))
    ## organize grid on Y axis
    #x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels])) #3

    ## switch X and Y axes
    #x4 = tf.transpose(x3, (0, 2, 1, 3))
    ## organize grid on X axis
    #x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels])) #3

    ## back to normal order (not combining with the next step for clarity)
    #x6 = tf.transpose(x5, (2, 1, 3, 0))

    ## to tf.image_summary order [batch_size, height, width, channels],
    ##   where in this case batch_size == 1
    #x7 = tf.transpose(x6, (3, 0, 1, 2))

    ## scale to [0, 255] and convert to uint8
    #return tf.image.convert_image_dtype(x7, dtype = tf.uint8) 
