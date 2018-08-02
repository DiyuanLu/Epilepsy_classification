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
import pickle

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
    data_dir = '../data/Whole_data/test_data/'
    #data_dir = 'test_files'
    files = find_files(data_dir, pattern='Data*.csv', withlabel=False)
    data_version = 'Data'
    '''band pass filtering'''
    for ind, filename in enumerate(files):
        data = read_data(filename, header=None, ifnorm=False)

        #lowcut = [1, 4, 8, 13, 30]
        #highcut = [4, 8, 13, 30, 70]
        lowcut = [0.5]
        highcut = [180]
        # Filter a noisy signal.  x, y, deltax, deltay, alphax, alphay, theatax, thetay, betax, betay, gammax, gammay
        filter_data = np.zeros((10240, 2))
        #ipdb.set_trace()
        #filter_data[:, 0:2] = data
        save_name = '../data/Whole_data/low_pass_filter/test/filter_{}'.format(os.path.basename(filename))
        #save_name = 'test_files/filter_{}'.format(os.path.basename(filename))
        for jj in range(len(lowcut)):
            
            y0 = butter_bandpass_filter(data[:, 0], lowcut[jj], highcut[jj], fs, order=3)
            y1 = butter_bandpass_filter(data[:, 1], lowcut[jj], highcut[jj], fs, order=3)
            filter_data[:,jj*2] = y0
            filter_data[:,jj*2+1] = y1 
        if ind % 2000 == 0:
            print(ind, filename)
            #for ii in range(2):
            plt.figure(figsize=(10, 8))
            plt.plot(filter_data[:, 0], 'c', label='filter-data1')
            plt.plot(data[:, 0], 'm', label='original-data1')
            plt.title('Low pass filtered data')
            plt.legend(loc='best')
            plt.savefig(save_name[0:-4]+'.png', format='png')
            plt.close()


        np.savetxt(save_name, filter_data, header='0.5~~180Hz_1, 0.5~180Hz_2', delimiter=',', comments='')
            #y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
            #plt.plot(y, label='Filtered signal ({}~{} Hz)'.format(lowcut[ind], highcut[ind]))
            #plt.xlabel('time (seconds)')
            ##plt.hlines([-a, a], 0, T, linestyles='--')
            ##plt.grid(True)
            #plt.axis('tight')
            #plt.legend(loc='upper left')
    #'''pickle data'''
    #for ind, filename in enumerate(files):
        ### pickle
        
        #try:
            #data_train_all = pickle.load(open('data/pickle_data_train_{}.p'.format(data_version), 'rb')) 
            #labels_train_all = pickle.load('data/pickle_labels_train_{}.pl'.format(data_version))
        #except:
            #num_train = len(files)
            ##data_train_all = np.zeros((num_train, 10240, 2))
            #data_train_all = []
            #labels_train_all = np.zeros((num_train))
            #for ind, filename in enumerate(files):
                ##data = read_data(filename, header=None, ifnorm=ifnorm)
                ##data_train_all[ind, :, :] = data
                #data_train_all.append(filename)
                #if 'F' in filename:
                    #labels_train_all[ind] = 1

            #pickle.dump(data_train_all, open( 'data/pickle_data_train{}.p'.format(data_version), 'wb'))
            #pickle.dump(labels_train_all, open( 'data/pickle_lables_train{}.p'.format(data_version), 'wb'))


run()
