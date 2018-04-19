#### util functions for EEG classification
##### file operations
##### data manipulation
##### fploting
import fnmatch
import numpy as np
import tensorflow as tf
import csv
import codecs
from scipy.signal import decimate
import multiprocessing
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import ipdb
import random
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

###################### files operation##########################
def find_files(directory, pattern='*.csv', withlabel=True):
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
    os.rename(filename, os.path.dirname(filename) + '/' + os.path.basename(filename)[5:])

def remove_files(filename):
    os.remove(filename)

def multiprocessing_func(data_dir):
    filenames = find_files(data_dir, pattern='Data*.csv', withlabel=False )
    pool = multiprocessing.Pool()
    version = 'downsampling'      #None#'remove'       # 'rename'      # 'rename'        # 
    if version == 'downsampling':
        for ds in [2]:
            pool.map(partial(downsampling, ds_factor=ds), filenames)
            print "Downsampling Done!"
    elif version == 'rename':
        pool.map(rename_files, filenames)
        print "rename Done!"
    elif version == 'remove':
        pool.map(remove_files, filenames)
        print "remove Done!"
    
    pool.close()
    
###################### Data munipulation##########################
def read_data(filename, ifaverage=False ):
    reader = csv.reader(codecs.open(filename, 'rb', 'utf-8'))
    x_data, y_data = np.array([]), np.array([])
    for ind, row in enumerate(reader):
        x, y = row
        x_data = np.append(x_data, x)
        y_data = np.append(y_data, y)
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    if ifaverage:
        data = np.expand_dims(np.mean(np.vstack((x_data, y_data)) , axis=0), axis=0 )  ### only get an average signal
    else:
        data = np.vstack((x_data, y_data))  ### read the pair
    return data.T   # output data shape (seq_len, channel)


def get_Data(data_dir, data_dir_test, batch_size=20, pattern='*.csv', withlabel=True):
    with tf.name_scope("Data"):
        #### Get data
        files_train = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
        files_test = find_files(data_dir_test, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
        file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
        file_tensor_test = tf.convert_to_tensor(files_test, dtype=tf.string)## convert to tensor
        dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_test = tf.data.Dataset.from_tensor_slices(file_tensor_test).repeat().batch(batch_size).shuffle(buffer_size=10000)
        ### map self-defined functions to the dataset
        dataset = dataset.map(get_data_and_label)
        dataset_test = dataset_test.map(get_data_and_label)
        ## create the iterator
        iter = dataset.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        ele_test = iter_test.get_next()   #you get the filename
        return ele, ele_test, iter, iter_test

#def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    #def _parse_line(line):
        #### decode the line into variables
        #x1, x2 = tf.decode_csv(line, [[0.], [0.]])   # each line with two values no headere
        #return x1, x2

   #dataset = (tf.data.TextLineDataset(file_path) # Read text file
       #.skip(1) # Skip header row
       #.map(decode_csv)) # Transform each elem by applying decode_csv fn
   #if perform_shuffle:
       ## Randomizes input using a window of 256 elements (read into memory)
       #dataset = dataset.shuffle(buffer_size=256)
   #dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   #dataset = dataset.batch(32)  # Batch size to use
   #iterator = dataset.make_one_shot_iterator()
   #batch_features, batch_labels = iterator.get_next()
   #return batch_features, batch_labels
   
#def my_input_func(_inputs):
    #'''_inputs is the (filename, '1'/'0')'''
        #_filename, _label = _inputs   # sentence and .wav file
        #def _parse_line(line):
            #### decode the line into variables
            #FIELD_DEFAULTS = [[0.0], [0.0]]
            #x1, x2 = tf.decode_csv(line, FIELD_DEFAULTS)   # each line with two values no headere
            #return x1, x2
        ## Processing
        #_spectrogram, _magnitude, _length = utils.get_spectrograms(_sound_file)
         
        #_spectrogram = utils.reduce_frames(_spectrogram, hp.win_length//hp.hop_length, hp.r)
        #_magnitude = utils.reduce_frames(_magnitude, hp.win_length//hp.hop_length, hp.r)

        #return _spectrogram, _magnitude, _length

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
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

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
    plt.savefig(save_name, format="png")
    plt.close()

def plot_smooth_shadow_curve(datas, colors=['darkcyan'], xlabel='training batches / 20', ylabel='accuracy', title='Loss during training', labels='accuracy_train', save_name="loss"):
    '''plot a smooth version of noisy data with mean and std as shadow
    data: a list of variables values, shape: (batches, )
    color: list of prefered colors
    '''
    plt.figure()
    fill_colors = ['lightcoral', 'plum']
    for ind, data in enumerate(datas) :
        data_smooth = smooth(data, window_len=25)
        data_smooth = data_smooth[0:len(data)]
        data_mean = np.mean(np.vstack((data, data_smooth)), axis=0)
        data_std = np.std(np.vstack((data, data_smooth)), axis=0)
        sizes = data_std.shape[0]
        plt.grid()
        plt.fill_between(np.arange(sizes), data_smooth - data_std, data_smooth + data_std, alpha=0.5, color=fill_colors[ind])
        plt.plot(np.arange(sizes), data_smooth, '-', linewidth=2, color=colors[ind], label=labels[ind])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(save_name, format="png")
    plt.close()

def plotdata(data, color='darkorchid', xlabel="training time", ylabel="loss", save_name="save"):
    '''
    data: 1D array '''
    plt.figure()
    plt.plot(data, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylabel == 'accuracy':
        plt.ylim([0.0, 1.05])
    elif ylabel == 'loss':
        plt.ylim([0.0, 1.0])
    plt.savefig(save_name + "_{}".format(ylabel))
    plt.close()
###################### Data munipulation##########################


if __name__ == "__main__":
    data_dir = ["data/train_data", "data/test_data"]
    for ind, dirr  in enumerate(data_dir):
        multiprocessing_func(dirr )
