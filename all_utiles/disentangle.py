### load data and compute approximate entropy and save
import numpy as np
import pandas as pd
import os
import ipdb
import fnmatch
from scipy.fftpack import fft
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


def sliding_PSD_slope(data, window=1024, stride=0.25*512, save_name='/results'):
    '''get windowed data with strides
    param:
        data: 1D array
    return:
        windowed_data:
        e.g. array([[ 0,  1,  2,  3,  4],
       [ 2,  3,  4,  5,  6],
       [ 4,  5,  6,  7,  8],
       [ 6,  7,  8,  9, 10],
       [ 8,  9, 10, 11, 12],
       [10, 11, 12, 13, 14],
       [12, 13, 14, 15, 16],
       [14, 15, 16, 17, 18]])
'''
    
    num_seg = np.int((data.size - np.int(window)) // np.int(stride) + 1)
    #expand_data = np.zeros((num_seg, window))
    shape = (num_seg, window)      ## done change the num_seq
    strides = (data.itemsize*stride, data.itemsize)
    expand_x = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    
    fft_seg = fft(expand_x, axis=1)
    PSD_seg = np.abs(fft_seg) ** 2
    scale_PSD = np.log(PSD_seg + 1e-9)
    from scipy.stats import linregress
    slopes = np.zeros((PSD_seg.shape[0]))
    for ii in range(PSD_seg.shape[0]):
        slope, intercept, r, p, std = linregress(scale_PSD[ii, 0:window//2], np.log(np.arange(window//2)+1))
        slopes[ii] = slope
    print(scale_PSD.shape)
    #plt.loglog(np.arange(window//2), PSD_seg[0:10, 0:window//2].T)
    #plt.show()
    #ipdb.set_trace()
    mask = np.ones(15) / 15.0
    result = np.convolve(slopes, mask, 'same')
    plt.figure(figsize=(15,8))
    plt.subplot(211)
    plt.plot(np.arange(data.size)/512.0, data)
    plt.xlim([0, data.size/512.0])
    plt.subplot(212)
    plt.plot(np.arange(slopes.size)/(slopes.size/(data.size/512.0)), slopes)
    plt.ylim([-0.5, 0.0])
    plt.xlim([0, slopes.size])
    plt.plot(np.arange(result.size)/(slopes.size/(data.size/512.0)), result, 'm')
    plt.xlim([0, result.size/(slopes.size/(data.size/512.0))])

    plt.savefig(save_name+"EI_ratio_window{}_stride{}.png".format(window, stride), format='png')
    
    plt.close()
    return slopes
    
    
data_dir = 'data/test_files'
save_name = "data/test_files/disentangle/EI_ratio/"
'''slide and seg'''

#seq_len = 10240
#width = 2

files_wlabel = find_files(data_dir, pattern='*.csv', withlabel=False)
num_files = len(files_wlabel)

#datas = np.zeros((100, seq_len, width))
#labels = np.array(files_wlabel)[:, 1].astype(np.int)
#filenames = np.array(files_wlabel)[:, 0].astype(np.str)
'''cluster'''
#files_wlabel = ['data/test_files/sep09_anno_8_test-long.csv', 'data/test_files/oct29_anno_5_pre_seizure.csv', 'data/test_files/oct29_anno_6_pre_seizure.csv']
#datas = np.zeros((num_files*10240))
#datas = np.zeros((num_files, 10240))
window = 128
for ind, filename in enumerate(files_wlabel):
    print(filename)
    #for window in range(16, 33, 14):
    data = read_data(filename, header=0, ifnorm=False)
    #datas[ind, :] = data[:, 0]

    #ipdb.set_trace()
    slopes = windowed_data = sliding_PSD_slope(data[:, 0], window=window, stride=np.int(0.75*window), save_name=save_name+os.path.basename(filename)[0:-4])  ##np.int(0.75*512)
    
    
#disentangel_ex_inhi(data, window=window, mode='average', save_name=save_name+os.path.basename(filename)[0:14])
#0 0 0 1 1 0 1 1 0 1 0 0 1 0 0 0 1 0 1 1 1 

print("Done")
    #ipdb.set_trace()
    #np.savetxt(save_name, aug_data, header="datax,datay,corrx,corry", delimiter=',', comments='')





