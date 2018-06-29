### load data and compute approximate entropy and save
import numpy as np
import pandas as pd
import os
import ipdb
import fnmatch
import matplotlib.pyplot as plt
from scipy.stats import zscore
import functions as func
from scipy.spatial.distance import euclidean
'''Usefull functions'''
def find_files(directory, pattern='Data*.txt', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                #if 'Data_F' in filename:
                    #label = '1'
                #elif 'Data_N' in filename:
                    #label = '0'
                if 'base' in filename:
                    label = '0'
                elif 'tip' in filename:
                    label = '1'
                elif 'seizure' in filename:
                    label = '2'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    #random.shuffle(files)   # randomly shuffle the files
    return files

def read_data(filename, header=None, ifnorm=True ):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''

    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm    ##data.shape (2048, 1)

    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]

    return data



def moving_opt_in_window(values, window=10, stride=5, mode='average'):
    if mode == 'average':
        mask = np.ones(window) / window
        result = np.convolve(values, mask, 'same')
    elif mode == 'sum':
        mask = np.ones(window)
        result = np.convolve(values, mask, 'same')

    return result

        
def disentangel_ex_inhi(data, window=10, mode='average', save_name='results/'):
    '''get exitation and inhibition from EEG'''
    data = data[:, 0]
    exi = data[1:] - data[0:-1]
    inhi = np.copy(exi)
    inhi[inhi>0] = 0
    exi[exi<=0] = 0
    inhi = inhi * (-1)

    new_exi = moving_opt_in_window(exi, window=window, mode="average")
    new_inhi = moving_opt_in_window(inhi, window=window, mode="average")
    #ipdb.set_trace()
    corr = np.corrcoef(new_exi, new_inhi)[0, 1]
    #integ_exi = np.trapz(new_exi)
    #integ_inhi = np.trapz(new_inhi)
    #dist = euclidean(new_exi, new_inhi)
    '''plt.figure(), plt.subplot(211), plt.plot(add_exi, label='exi'), plt.plot(add_inhi, label='inhi'), plt.legend(loc='best'), plt.subplot(212), plt.plot(integ_exi, label='exi'), plt.plot(integ_inhi, label='inhi')'''
    #ipdb.set_trace()
    cum_exi = np.cumsum(new_exi)
    #cum_rate_exi = moving_opt_in_window(cum_exi, window=window, mode='average')
    ee = np.cumsum(cum_exi[0:119400].reshape(-1, 20), axis=1) / (1 + np.arange(20))
    cum_inhi = np.cumsum(new_inhi)
    ii = np.cumsum(cum_inhi[0:119400].reshape(-1, 20), axis=1) / (1 + np.arange(20))
    
    np.cumsum(aa.reshape(-1, 5), axis=1) / (1 + np.arange(5))
    add_exi = moving_opt_in_window(new_exi, window=window, mode='sum')
    add_inhi = moving_opt_in_window(new_inhi, window=window, mode='sum')


    plt.figure()
    plt.subplot(311)
    plt.title('Disentangle exi and inhi')
    plt.plot(data, 'b', label='original data')
    plt.xlim([0, data.size])
    plt.legend()
    plt.subplot(312)
    plt.title("Exci and inhi within window {}".format(window))
    plt.plot(new_exi, 'm', label='exci')
    plt.plot(new_inhi, 'c', label='inhi')
    plt.xlim([0, data.size])
    plt.legend(loc='best')
    plt.subplot(313)
    plt.plot(add_exi, 'm', label='exci')
    plt.plot(add_inhi, 'c', label='inhi')
    plt.title("Accumulated activity within window {}".format(window))
    plt.xlim([0, data.size])
    plt.legend(loc='best')
    plt.xlabel("time steps corr={}".format(corr))
    #plt.ylabel("accumulated activity difference")
    #ipdb.set_trace()
    plt.savefig(save_name+"-Disentangle-cum{}.png".format(window), format='png')
    plt.close()

    plt.figure()
    plt.subplot(311)
    plt.title('Disentangle exi and inhi change rate')
    plt.plot(data, 'b')
    plt.subplot(312)
    plt.title('')
    plt.plot(cum_exi / (1+np.arange(cum_exi.size)), 'm', label='exci')
    plt.plot(cum_inhi / (1+np.arange(cum_inhi.size)), 'c', label='inhi')
    plt.subplot(313)
    plt.plot(cum_exi / (1+np.arange(cum_exi.size)), 'm', label='exci')
    plt.plot(cum_inhi / (1+np.arange(cum_inhi.size)), 'c', label='inhi')
    
    
data_dir = 'data/test_files/anno'
save_name = "data/test_files/anno/disentangle/"
'''slide and seg'''

#seq_len = 10240
#width = 2

#files_wlabel = find_files(data_dir, pattern='*.csv', withlabel=False)
#num_files = len(files_wlabel)

#datas = np.zeros((100, seq_len, width))
#labels = np.array(files_wlabel)[:, 1].astype(np.int)
#filenames = np.array(files_wlabel)[:, 0].astype(np.str)
'''cluster'''
files_wlabel = ['data/test_files/sep09_anno_8_test-long.csv']
for ind, filename in enumerate(files_wlabel):
    print(filename)
    for window in range(16, 33, 14):
        data = read_data(filename, header=1, ifnorm=False)
        #datas[ind, :, :] = data
        
        disentangel_ex_inhi(data, window=window, mode='average', save_name=save_name+os.path.basename(filename)[0:14])
        

    print("Done")
    ipdb.set_trace()
    #np.savetxt(save_name, aug_data, header="datax,datay,corrx,corry", delimiter=',', comments='')





