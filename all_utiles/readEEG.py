#### read EEG recording and plot it

import numpy as np
from scipy import fft as scifft
from scipy import signal
import fnmatch
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#from scipy.io import wavfile
import ipdb
import random
import pandas as pd

#import functions as func
#import pyeeg
import matplotlib.pylab as pylab
params = {'legend.fontsize': 12,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)


def autocorrelation(x) :
    """
    Compute the autocorrelation of the signal, based on the properties of the
    power spectral density of the signal.
    """
    xp = x-np.mean(x)
    f = np.fft.fft(xp)
    p = np.array([np.real(v)**2+np.imag(v)**2 for v in f])
    pi = np.fft.ifft(p)
    return f[:x.size/2], np.real(pi)[:x.size/2]/np.sum(xp**2)

def lag1_ar(data, window=2048, lag=1) :
    """
    data:  1D array, the whole data
    Compute the autocorrelation of the signal by defination
    https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781783553358/7/ch07lvl1sec75/autocorrelation
    return:
    the alg1 correlation coefficient given the lag and window size"""
    lag_1 = []
    for ii in range(data.size-window-lag):
        ar1 = np.corrcoef(np.array([data[ii:window+ii], data[ii+lag:window+ii+lag]]))
        lag_1.append(ar1[0, 1])
    return lag_1

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
                elif 'Bonn' in filename:
                    if 'Z' in filename:
                        label = '0'
                    elif 'O' in filename:
                        label = '0'
                    elif 'N' in filename:
                        label = '1'
                    elif 'F' in filename:
                        label = '1'
                    elif 'S' in filename:
                        label = '2'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    print(len(files))
    random.shuffle(files)
    return files 

    
def read_data(filename, header=None, ifnorm=True):
    from scipy.stats import zscore
    import pandas as pd
    '''read data from .csv
    Param:
        filename: string e.g. 'data/Data_F_Ind0001.csv'
        ifnorm: boolean, 1 zscore normalization
        start: with filter augmented data, start index of the column indicate which group of data to use
        width: how many columns of data to use, times of 2
    return:
        data: 2d array [seq_len, channel]'''

    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data

def get_var(data):
    variance = np.var(data, 1)

    return variance

def plotSpectrum(data,Fs, color='c', label='x'):
    """
    Plots a Single-Sided Amplitude Spectrum of data(t)
    """
    n = len(data) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scifft(data)/n # fft computing and normalization
    Y = Y[range(n/2)]

    plt.plot(frq,np.abs(Y), color, label=label) # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')

def plot_spectrogram(data, fs=512):
    spec = plt.specgram(data, cmap='viridis', NFFT=256, Fs=fs)
    (Spec, f, t) = spec[0], spec[1], spec[2]             
    plt.title('Spectrogram of orginal signal')
    plt.xlabel('time / s')
    plt.ylabel('frequency')
    plt.xlim([0, t[-1]])
    plt.ylim([0, f[-1]])


                    
def plotPowerSpectrum(data, Fs, color='m', label='x', title='Power spectral density'):
    f, Pxx_den = signal.welch(data[:, 0], Fs, window='hanning', nperseg=512, noverlap=128)
    #plt.semilogy(f, Pxx_den, color, label=label)
    plt.plot(f, Pxx_den, color, label=label)
    plt.title(title)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('power spectrum density')

def plotWaveletSpectrum(data, color='m', label='x', title='Power spectral density'):
    cwtmatr = signal.cwt(data, signal.morlet, np.arange(1, 25))
    plt.imshow(cwtmatr, cmap='jet', aspect="auto")
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title("Wavelet spectrogram")
    plt.colorbar()

def plotOnePair(data11, data12, data21, data22):
    '''plot the original data-pair'''
    gs_top = plt.GridSpec(4, 1, hspace=0)
    gs_base = plt.GridSpec(4, 1)
    fig = plt.figure()
    ###fig.add_subplot(gs_base[0])
    ###plt.plot(np.average(np.vstack((data1, data2)), axis=0), 'orchid', label='average')
    ###plt.title("Average")
    ## share x axis
    ax = fig.add_subplot(gs_top[0, :]) # Need to create the first one to share...
    ax.plot(data11, 'c', label="non-focal")
    ax.plot(data21, 'blueviolet', label="focal")
    plt.legend()
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel("recording/ mV")
    other_axes = [fig.add_subplot(gs_top[i,:], sharex=ax) for i in range(1, 2)]
    other_axes[0].plot(data12, 'c', label="non-focal")
    plt.plot(data22, 'blueviolet', label="focal")
    plt.legend()
    plt.ylabel("recording/ mV")
    plt.xlim([0, 10240])

    fig.add_subplot(gs_base[2])
    plotPowerSpectrum(data11, 512, color= 'c', label="non-focal")
    plotPowerSpectrum(data21, 512, color= 'blueviolet' , label="focal")
    plt.title("power spectrum")
    plt.legend()
    plt.xlim([0, 20])
    #plt.ylim([0.01, 10000])
    
    fig.add_subplot(gs_base[3])
    plotSpectrum(data12, 512, color= 'c', label="non-focal")
    plotSpectrum(data22, 512, color= 'blueviolet' , label="focal")
    plt.title("spectrum")
    plt.legend()
    plt.xlim([0, 100])
    plt.tight_layout()


def get_features(data, fs=512):
    features = []
    pfd = pyeeg.pfd(data)
    hfd = pyeeg.hfd(data)
    hjorth = pyeeg.hjorth(data)
    specentro = pyeeg.spectral_entropy(data)
    svdentro = pyeeg.svd_entropy(data)
    fisher = pyeeg.fisher_info(data)
    apen= pyeeg.ap_entropy(data)
    dfa= pyeeg.dfa(data)
    hurst= pyeeg.hurst(data)

    return

def get_normalized_freq_band_PSD_save_csv(datas, labels, Fs=512, save_name='data/'):
    '''Get normalized PSD in different frequency bands
    param:
        data: batch_size*seq_len_channels
    '''
    freq_bands = [0, 4, 8, 15, 30, 100, 512]
    num_bands = 6
    norm_band_psd = np.zeros((datas.shape[0], num_bands * 2 + 1))
    
    norm_band_psd[:, 0] = labels

    #plt.figure()
    for ind in range(datas.shape[0]):
        for channel in range(2):
            f, Pxx_den = signal.welch(datas[ind, :, channel], Fs, window='hanning', nperseg=512, noverlap=128)
        
            band_psd = np.split(Pxx_den, freq_bands[1:-1])
            band_psd_sum = [np.sum(band_psd[i]) for i in range(len(band_psd))]
            norm_psd = band_psd_sum / np.sum(Pxx_den)
            norm_band_psd[ind, channel*6+1: (channel+1)*6+1] = np.array(norm_psd)

        #if labels[ind] == 0:
            #plt.plot(norm_psd, 'c')
        #else:
            #plt.plot(norm_psd, 'm')
    

    #plt.show()
    np.savetxt(save_name+'band_PSD_test.csv', norm_band_psd.reshape(datas.shape[0], -1), header='#'+np.str(freq_bands)+'x, y', delimiter=',', comments='')

    print("Done!")
    

def get_corr_len(datas, labels, save_name='results/', postfix='Nonfocal'):
    '''based on the method in https://ieeexplore.ieee.org/document/7727334/
    get the distibution of correlation length to determine the proper length to segment
    Param:
        datas: batch_size*seq_len*width*channel
    return:
    '''
    corr_len = np.zeros((datas.shape[0]))
    auto_corr = []
    for ind in range(datas.shape[0]):
        if ind % 500 == 0:
            print("file: ", ind)
        err = datas[ind, :, 0] - np.mean(datas[ind, :, 0])
        variance = np.sum(err ** 2) / datas[ind, :, 0].size
        correlated = np.correlate(err, err, mode='full')/variance
        correlated = correlated[correlated.size//2:]
        sign = np.sign(correlated)
        signchange = np.where(sign[1:] - sign[0:-1])[0][0]   ### get where the line cross the zero line
        corr_len[ind] = signchange / 173.6
        auto_corr.append(correlated)

    np.savetxt('corr_len_distribution_{}.csv'.format(postfix), corr_len, header='#correlation length distribution', delimiter=',', comments='')
    plt.hist(corr_len, color='slateblue', bins=100, alpha=0.8)  # arguments are passed to np.histogram
    plt.title("Correlation length distribution")
    plt.xlabel("time delay (s)")
    plt.savefig("Correlation length distribution-{}".format(postfix), format='png')
    plt.close()
    return corr_len



#######################################################################################
data_dir = "../data/Bonn_data/"

#data_dir = "test_files/"
files = find_files(data_dir, pattern="Bonn*.csv", withlabel=True)

datas = np.zeros((len(files), 4097+1))
labels = np.array(files)[:, 1].astype(np.int)
filenames = np.array(files)[:, 0].astype(np.str)
datas[:, 0] = labels

for ind, filename in enumerate(filenames):
    data = read_data(filename, header=None, ifnorm=True)
    datas[ind,1:] = np.squeeze(data)
ipdb.set_trace()
np.savetxt('Bonn_all_shuffle_data.csv', datas, header=['lable']+['value']*4097, delimiter=',', comments='')
### get the normalized PSD in frequency bands
#get_normalized_freq_band_PSD_save_csv(datas, labels, Fs=512, save_name=data_dir)





#plotPowerSpectrum(data, 50, color='m', label='x', title='PSD {}'.format(filename))
#plt.show()
#x_data2, y_data2 = read_data(files[1])
##fft, correlation = autocorrelation(x_data)

##sd.play(x_data, 1000)
##wavfile.write("results/audio/" + filename + "fs1000.wav", 1000, x_data)
#ipdb.set_trace()
#plotOnePair(x_data1, y_data1, x_data2, y_data2)
#ipdb.set_trace()
##plt.show()
#plt.savefig(  files[0][0:-4] +"compare_F_NF_linear.png")
#plt.close()


