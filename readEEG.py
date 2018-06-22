#### read EEG recording and plot it

import numpy as np
from scipy import fft as scifft
from scipy import signal
import fnmatch
import matplotlib.pyplot as plt
#from scipy.io import wavfile
import ipdb
import os
import pandas as pd

#import functions as func
#import pyeeg


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

def find_files(directory, pattern='D*.csv', withlabel=True):
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



def plotAllPairs(files):
    X_DATA, Y_DATA = np.array([])

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


    
data_dir = "test_files"
files = find_files(data_dir, pattern='Data*.csv', withlabel=False)
plt.figure()
window = 1024
lag = 1
#for window in range(1024, 5121, 512):
    ##for lag in range(1, 52, 2):
        ##for ind, filename in enumerate(files):
            ##print filename
            ##data_x, data_y = read_data(filename)
            ##corr = lag1_ar(data_x, window=window, lag=lag)
            ##if "_F_" in filename:
                ##color = "b"
            ##else:
                ##color= 'c'
            ##plt.plot(corr, color=color, label='{}_ind{}'.format(filename[21:23], ind))
        ##plt.title("AR(1)")
        ##plt.legend(loc="best")
        ##plt.xlabel("time step/ win={}, lag={}".format(window, lag))
        ##plt.savefig("data/test_files/AR1_win{}_lag{}".format(window, lag))
        ##plt.close()
focal = np.zeros((4, 10240, 2))
nonfocal = np.zeros((4, 10240, 2))
fcount=0
ncount = 0
for filename in files:
    print( filename)
    data = read_data(filename, header=None, ifnorm=True)
    if 'F' in filename:
        focal[fcount, :, :] = data
        fcount += 1
    elif 'N' in filename:
        nonfocal[ncount, :, :] = data
        ncount += 1
        
ipdb.set_trace()
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


