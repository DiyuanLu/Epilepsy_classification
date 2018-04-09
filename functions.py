#### util functions for EEG classification
import fnmatch
import numpy as np
import csv
import codecs
from scipy.signal import decimate
import multiprocessing
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
#import ipdb
import random


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

def read_data(filename ):
    reader = csv.reader(codecs.open(filename, 'rb', 'utf-8'))
    x_data, y_data = np.array([]), np.array([])
    for ind, row in enumerate(reader):
        x, y = row
        x_data = np.append(x_data, x)
        y_data = np.append(y_data, y)
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    return x_data, y_data

def downsampling(filename, ds_factor):
    """Downsample the original data by the factor of ds_factor to get a low resolution version"""
    x, y = read_data(filename)
    ds_x =  decimate(x, ds_factor)
    ds_y =  decimate(y, ds_factor)
    np.savetxt(os.path.splitext(filename)[0] + "ds_" + np.str(ds_factor) + ".csv", zip(ds_x, ds_y), delimiter=',', fmt="%10.5f")

def rename_files(files):
    os.rename(files, os.path.splitext(files)[0] + '.csv')

def remove_files(files):
    os.remove(files)

def multiprocessing_func(data_dir):
    files_train = find_files(data_dir, pattern='*16ds_8.csv', withlabel=False )
    pool = multiprocessing.Pool()
    # for ds in [8]:
    #     pool.map(partial(downsampling, ds_factor=ds), files_train)
    #     print "Done!"
    # print files_train
    pool.map(remove_files, files_train)
    pool.close()

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

def plot_learning_curve(train_scores, test_scores, title="Learning curve", save_name="learning curve"):
    '''plot smooth learning curve
    train_scores: n_samples * n_features
    test_scores: n_samples * n_features
    '''
    plt.figure()
    plt.title(title)
    train_sizes = len(train_scores)
    plt.xlabel("training batches")
    plt.ylabel("accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(np.arange(train_sizes), train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.25,color="m")
    plt.fill_between(np.arange(train_sizes), test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.25, color="c")
    plt.plot(np.arange(train_sizes), train_scores_mean, '-', color="m",
             label="Training score")
    plt.plot(np.arange(train_sizes), test_scores_mean, '-', color="c",
             label="Test score")

    plt.legend(loc="best")
    plt.savefig(save_name, format="png")
    plt.close()

def plot_smooth_shadow_curve(data, title="Loss during training", save_name="loss"):
    '''plot a smooth version of noisy data with mean and std as shadow
    data: n_samples * n_trials'''
    data_smooth = smooth(data, window=21)
    data_smooth = data_smooth.shape[0]
    #data_mean = np.mean(data, axis=1)
    #data_std = np.std(data, axis=1)
    sizes = data.shape[0]
    plt.grid()
    plt.fill_between(np.arange(sizes), data_smooth - data_std, data_smooth + data_std, alpha=0.3, color="c")
    plt.plot(np.arange(sizes), data_smooth, '-', color="b")
    plt.title(title)
    plt.ylim([0.0, 2.])
    plt.savefig(save_name, format="png")
    plt.close()

def plotdata(data, color='c', xlabel="training time", ylabel="loss", save_name="save"):
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

def save_data(data, save_name="save_data"):
    '''save data into a .csv file'''

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

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
#if __name__ == "__main__":
    #data_dir = "data/train_data"
    #multiprocessing_func(data_dir)
