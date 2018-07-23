from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
import os
import ipdb
import fnmatch
import matplotlib.pyplot as plt
from scipy.signal import freqz
import random
from scipy.stats import zscore


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_files(directory, pattern='Data*.csv', withlabel=True):
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

def read_data(filename, header=None, ifnorm=True ):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''
    
    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data



def run():
    

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 512
    data_dir = 'data/train_data'
    #data_dir = 'test_files'
    files = find_files(data_dir, pattern='Data*.csv', withlabel=False)
    
    for ind, filename in enumerate(files):
        data = read_data(filename, header=None, ifnorm=False)

        lowcut = [1, 4, 8, 13, 30]
        highcut = [4, 8, 13, 30, 70]
        # Filter a noisy signal.  x, y, deltax, deltay, alphax, alphay, theatax, thetay, betax, betay, gammax, gammay
        filter_data = np.zeros((10240, 2*(len(lowcut)+1)))
        #ipdb.set_trace()
        filter_data[:, 0:2] = data
        save_name = 'data/train_data/filter/filter_{}'.format(os.path.basename(filename))
        #save_name = 'test_files/filter_{}'.format(os.path.basename(filename))
        for jj in range(1, len(lowcut)+1):
            
            y0 = butter_bandpass_filter(data[:, 0], lowcut[jj-1], highcut[jj-1], fs, order=3)
            y1 = butter_bandpass_filter(data[:, 1], lowcut[jj-1], highcut[jj-1], fs, order=3)
            filter_data[:,jj*2] = y0
            filter_data[:,jj*2+1] = y1 
        if ind % 1000 == 0:
            print(ind, filename)
            for ii in range(6):
                plt.plot(filter_data[:, ii*2], label='data1-{}'.format(ii))
            plt.title('Band pass filtered data')
            plt.legend(loc='best')
            plt.savefig(save_name[0:-4]+'.png', format='png')
            plt.close()


        np.savetxt(save_name, filter_data, header='original_1,original_2,1-4Hz_1,1-4Hz_2,4-8Hz_1,4-8Hz_2,8-13Hz_1,8-13Hz2,13-30Hz_1,13-30Hz_2,30-70Hz_1,30-70Hz_2', delimiter=',', comments='')
            #y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
            #plt.plot(y, label='Filtered signal ({}~{} Hz)'.format(lowcut[ind], highcut[ind]))
            #plt.xlabel('time (seconds)')
            ##plt.hlines([-a, a], 0, T, linestyles='--')
            ##plt.grid(True)
            #plt.axis('tight')
            #plt.legend(loc='upper left')

    #plt.show()


run()