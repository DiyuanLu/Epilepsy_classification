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
import matplotlib.pylab as pylab
from scipy.stats import zscore
import pandas as pd

# import scipy.stats as stats
params = {'legend.fontsize': 12,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
import matplotlib

###################### files operation##########################
def find_files(directory, pattern='Data*.txt', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'Data_F' in filename:
                    label = '1'
                elif 'Data_N' in filename:
                    label = '0'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    random.shuffle(files)   # randomly shuffle the files
    return files

def rename_files(filename):
    #os.rename(filename, os.path.dirname(filename) + '/Data_' + os.path.basename(filename)[5:])
    os.rename(filename, filename[0:-5] + ".csv")

def remove_files(filename):
    os.remove(filename)

def multiprocessing_func(data_dir):
    '''PicklingError: Can't pickle <type 'function'>: attribute lookup __builtin__.function failed
    PLEASE disable the import ipdb!!!'''
    filenames = find_files(data_dir, pattern='Data*.csv', withlabel=False )
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

    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values[:, start:start+width]   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data

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

def load_and_save_data(data_dir, pattern='Data*.csv', withlabel=True, ifnorm=True, num_classes=2):
    '''Keras way of loading data
    return: x_train, y_train, x_test, y_test
        x_train: [num_samples, seq_len, channel]
        y_train: [num_samples, ]  ## int label
        x_train: [num_samples, seq_len, channel]
        x_train: [num_samples, ]  ## int label
        '''
    #### Get data
    files_wlabel_train = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
    # files_wlabel_test = find_files(data_dir_test, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
    # ipdb.set_trace()
    files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
    # files_test, labels_test = np.array(files_wlabel_test)[:, 0], np.array(files_wlabel_test)[:, 1].astype(np.int)   ##
    data_train = np.zeros([len(files_train), 10240, 4])
    for ind in range(len(files_train)):
        if ind % 20 == 0:
            print("train", ind, 'files_train', files_train[ind], 'label', labels_train[ind])
        data = read_data(files_train[ind], ifnorm=ifnorm)
        # ipdb.set_trace()
        data_train[ind] = data
    ipdb.set_trace()

    # data_test = []
    # for ind in range(len(files_test)):
    #     if ind % 20 == 0:
    #         print("test", ind, 'files_test', files_test[ind], 'label', labels_test[ind])
    #     data = read_data(files_test[ind], ifnorm=ifnorm)
    #     data_test.append(data)

    #np.savetxt(save_name, data, header=header, delimiter=',', fmt="%10.5f", comments='')
    np.savez("ori_aug_20test", data=data_train, label=np.array(labels_train))
    ####
    # return np.array(data_train), np.array(labels_train)

#x_train=xx_train, y_train=yy_train, x_test=xx_test, y_test=yy_test


def downsampling(filename, ds_factor):
    """Downsample the original data by the factor of ds_factor to get a low resolution version"""
    x, y = read_data(filename)
    ds_x =  decimate(x, ds_factor)
    ds_y =  decimate(y, ds_factor)
    np.savetxt(os.path.dirname(filename) + "/ds_" + np.str(ds_factor)  + os.path.basename(filename) , zip(ds_x, ds_y), delimiter=',', fmt="%10.5f")

def save_data(data, header='data', save_name="save_data"):
    '''save data into a .csv file
    data: list of data that need to be saved, (x1, x2, x3...)
    header: String that will be written at the beginning of the file.'''
    np.savetxt(save_name, data, header=header, delimiter=',', fmt="%10.5f", comments='')

def load_data(data_dir):
    '''Load variables' data from pre-saved .csv file
    return a dict '''
    reader = csv.reader(codecs.open(data_dir, 'rb', 'utf-8'))
    data = dict()
    for ind, row in enumerate(reader):
        if  ind == 0:
            names = row
        else:
            data[names[ind-1]]= np.array(row).astype(np.float32)
    return data, names

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
        
    num_seg = (data_x.shape[1] - np.int(window)) // stride + 1
    expand_data = np.zeros((data_x.shape[0], num_seg, window, data_x.shape[-1]))
    # ipdb.set_trace()
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
    lag_1 = np.zeros((data.size))
    for ii in range(data.size-window):  ###1000-200
        ar1 = np.corrcoef(np.array([data[ii:window+ii], data[ii+lag:window+ii+lag]]))
        if ii == 0:
            lag_1[0:window+1] = ar1[0, 1]
        else:
            lag_1[window+ii] = ar1[0, 1]
    return lag_1

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
'''('files_wlabel', ('data/train_data/Data_N_Ind_1_750/Data_N_Ind0300.csv', '0'))
('resi', 0, 'layer', 0, 'net', TensorShape([Dimension(None), Dimension(1024), Dimension(2), Dimension(8)]))
('resi', 0, 'layer', 1, 'net', TensorShape([Dimension(None), Dimension(512), Dimension(2), Dimension(16)]))
('resi', 0, 'layer', 2, 'net', TensorShape([Dimension(None), Dimension(256), Dimension(2), Dimension(32)]))
('resi net ', TensorShape([Dimension(None), Dimension(128), Dimension(2), Dimension(32)]))
('resi', 1, 'layer', 0, 'net', TensorShape([Dimension(None), Dimension(128), Dimension(2), Dimension(8)]))
('resi', 1, 'layer', 1, 'net', TensorShape([Dimension(None), Dimension(64), Dimension(2), Dimension(16)]))
('resi', 1, 'layer', 2, 'net', TensorShape([Dimension(None), Dimension(32), Dimension(2), Dimension(32)]))
('resi net ', TensorShape([Dimension(None), Dimension(16), Dimension(2), Dimension(32)]))
('resi', 2, 'layer', 0, 'net', TensorShape([Dimension(None), Dimension(16), Dimension(2), Dimension(8)]))
('resi', 2, 'layer', 1, 'net', TensorShape([Dimension(None), Dimension(16), Dimension(2), Dimension(16)]))
('resi', 2, 'layer', 2, 'net', TensorShape([Dimension(None), Dimension(16), Dimension(2), Dimension(32)]))
('resi net ', TensorShape([Dimension(None), Dimension(16), Dimension(2), Dimension(32)]))
'''

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

def ApEn(U, m, r):
    '''Pincus
    [13] suggested that m be 1 or 2, and r be
    0.1SD to 0.25SD ( SD isthe standard deviation of the
    data),'''
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m + 1) - _phi(m))
    
###################### plots ##########################
def PCA_plot(pca_fit):
    traces = []
    for name in ('Focal', 'Non-focal'):
        trace = Scatter(
            x=pca_fit[y==name,0],
            y=pca_fit[y==name,1],
            mode='markers',
            name=name,
            marker=Marker(
                size=12,
                line=Line(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5),
                opacity=0.8))
        traces.append(trace)

        data = Data(traces)
        layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                        yaxis=YAxis(title='PC2', showline=False))
        fig = plt.figure(data=data, layout=layout)
        plt.plot(fig)

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

def plot_smooth_shadow_curve(datas, ifsmooth=False, window_len=25, colors=['darkcyan'], xlabel='training batches / 20', ylabel='accuracy', title='Loss during training', labels='accuracy_train', save_name="loss"):
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
    else:
        for ind, data in enumerate(datas) :
            plt.plot(data, '-', linewidth=2, color=colors[ind], label=labels[ind])
            
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim([0, 1.05])
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(save_name+'.png', format="png")
    plt.close()

def plotdata(data, color='darkorchid', xlabel="training time", ylabel="loss", save_name="save"):
    '''
    data: 1D array '''
    plt.figure()
    plt.plot(np.arange(data.size)/512.0, data, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylabel == 'accuracy':
        plt.ylim([0.0, 1.05])
    elif ylabel == 'loss':
        plt.ylim([0.0, 1.0])
    plt.savefig(save_name + "_{}".format(ylabel))
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


def vis_layer_activation(layer_name, inputs, save_name='results/'):
    '''
    tensor_name: tensor_name of a specific layer
    return:
        activtion of the layer given the inputs and reuse the weights
    '''
    with tf.variable_scope(layer_name, reuse=True):
        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)

    weights = vars_lsit[0]
    biass = vars_lsit[1]

        
def visualize_fc_layer_activation(sess, layer_name, inputs, save_name='results/'):
    '''visualize fc layer activation given some inputs
    param: 
        sess: current session
        layer_name: the layer you want to visualize
        inputs: 2D array [batch_size, seq_len, width]
        
    return:
        activations: activations from each layer'''
    ipdb.set_trace()
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
    #for ind in range(inputs.shape[0]):
    
        #acti = sess.run(activation, feed_dict={example: inputs[ind, :, :]})
        #acti_tot[ind, :] = acti

    

def vis_conv_layer_activation(sess, layer_name, inputs, save_name='results/'):

    with tf.variable_scope(layer_name, reuse=True):
        vars_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
    # with tf.variable_scope(layer_name, reuse=True):vars =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name)
    ipdb.set_trace()
    activation = tf.nn.relu(tf.matmul(inputs, weights) + bias)
    
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[:,0,0,i], interpolation="nearest", cmap="gray")

    plt.savefig(save_name + "conv_kernals.pdf", format="pdf")
    plt.close()
    
def plot_train_samples(samples, true_labels, save_name='results/'):
    plt.figure()
    for ii in range(20):
        ax1 = plt.subplot(5, 4, ii +1)
        plt.plot(np.arange(samples[ii, :, 0])/ 512.0, samples[ii, :, 0])
        plt.xlim([0, samples[ii, :, 0]/ 512.0])
        plt.xlabel("label: "+ np.str(true_labels[ii]))
        plt.ylabel("Voltage (mV)")
        #plt.setp(ax1.get_yticklabels(), visible = False)
        plt.setp(ax1.get_xticklabels(), visible = False)
    plt.tight_layout()
    plt.savefig(save_name + 'samples_train.png', format = 'pdf')
    plt.close()

def plot_BB_training_examples(samples, true_labels, save_name='results/'):
    
    for ii in range(6):
        plt.figure()
        for ind in range(samples[ii, :, :].shape[-1]):
        
            ax1 = plt.subplot(samples[ii, :, :].shape[-1], 1, ind+1)
            plt.plot(np.arange(samples[ii, :, ind])/ 512.0, samples[ii, :, ind], label="data_{}".format(ind+1))
            plt.ylabel("Voltage (mV)")
            plt.xlabel("data samples, label={}".format(true_labels[ii]))
            plt.legend()
            plt.xlim([0, samples[ii, :, 0].size/ 512.0])
        plt.tight_layout()
        plt.savefig(save_name + "vis_train_data{}.png".format(ii), format="png")
        plt.close()

def plotbhSNE(x_data, label, window=512, num_classes=2, title="t-SNE", save_name='/results'):
    
    '''tSNE visualize the original setmented data
    Param:
        x_data: shape(batch, seq_len, width)''' #classify based on the activity, x_data = stateX
    # perform t-SNE embedding--2D plot
    vis_data = bh_sne(x_data, pca_d=3)  #steps*2
    # plot the result
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    ##plot the result
    plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("cool", num_classes))   ##
    plt.title("t-SNE in orginal {}-long segments".format(window))
    plt.colorbar(ticks=range(num_classes))
    plt.clim(-0.5, num_classes-0.5)
    plt.savefig(save_name+"t-SNE-{}.png".format(window), format='png')
    plt.close()
    
#def plotOnePair(data):
    #'''plot the original data-pair'''
 


####################### Data munipulation##########################

## #
#if __name__ == "__main__":
 ##     #data_dir = "data/train_data"
 ##     # data_dir_test = "data/test_data"
 ##     #data_dir = 'data/test_files'
 ##     # # read_data_save_tfrecord(data_dir)
    ## ddd = ["data/train_data", "data/test_data"]
     ## ipdb.set_trace()
     ## augment_data_with_ar1(filename)
    ## for direc in ddd:
    #multiprocessing_func("data/train_data")
 ##     # get_Data(data_dir, data_dir_test, pattern='Data*.csv', withlabel=True)
 ##     # multiprocessing_func(data_dir)
 ##     # read_from_tfrecord("data/test_files/test_files.tfrecords")
 ##     #read_tfrecord()
 ##     #read_data_save_one_csv(data_dir_test)
 ##     #filename = "data/train_data/train_data.csv"
 ##     #read_data(filename)
     ## data_dir = "data/test_data"
     ## load_and_save_data(data_dir, pattern='*_aug2.csv', withlabel=True, num_classes=2)
