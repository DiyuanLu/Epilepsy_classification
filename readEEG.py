#### read EEG recording and plot it

import numpy as np
from scipy import fft as scifft
from scipy import signal
import csv
import codecs
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io import wavfile
import ipdb
import os
import functions as func

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

def get_var(data):
    variance = np.var(data, 1)

    return variance

def plotSpectrum(data,Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of data(t)
    """
    Fs = 400
    n = len(data) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    Y = scifft(data)/n # fft computing and normalization
    Y = Y[range(n/2)]

    plt.plot(frq,np.abs(Y),'orchid') # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')

def plotPowerSpectrum(data, Fs):
    f, Pxx_den = signal.periodogram(data, Fs)
    plt.semilogy(f, Pxx_den, 'royalblue')
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')

def plotWaveletSpectrum(data):
    cwtmatr = signal.cwt(data, signal.morlet, np.arange(1, 25))
    plt.imshow(cwtmatr, cmap='jet', aspect="auto")
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title("Wavelet spectrogram")
    plt.colorbar()

def plotOnePair(data1, data2):
    '''plot the original data-pair'''
    gs_top = plt.GridSpec(4, 1, hspace=0)
    gs_base = plt.GridSpec(4, 1)
    fig = plt.figure()
    ### share x axis
    ax = fig.add_subplot(gs_top[0, :]) # Need to create the first one to share...
    ax.plot(data1, 'orchid', label="data_1")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.ylabel("recording/ mV")
    other_axes = [fig.add_subplot(gs_top[i,:], sharex=ax) for i in range(1, 2)]
    other_axes[0].plot(data2, 'blueviolet', label="data_2")
    plt.ylabel("recording/ mV")
    plt.xlim([0, 10240])

    fig.add_subplot(gs_base[2])
    plotPowerSpectrum(data1, 1000)
    plt.title("power spectrum")
    plt.xlim([0, 100])
    plt.ylim([0.01, 10000])
    
    fig.add_subplot(gs_base[3])
    plotSpectrum(data1,100)
    plt.title("spectrum")
    plt.xlim([0, 100])
    plt.tight_layout()
    #plt.subplot(414)
    #plotWaveletSpectrum(data1)


def plotAllPairs(files):
    X_DATA, Y_DATA = np.array([])

data_dir = "data/test_files/"
files = func.find_files(data_dir, withlabel=False)
for filename in files:
    print filename
    x_data, y_data = read_data(filename)
    #fft, correlation = autocorrelation(x_data)

    #sd.play(x_data, 1000)
    #wavfile.write("results/audio/" + filename + "fs1000.wav", 1000, x_data)
    #ipdb.set_trace()
    plotOnePair(x_data, y_data)
    plt.savefig( filename[0:-4] +"_original.png")
    plt.close()

