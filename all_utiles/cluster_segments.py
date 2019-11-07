### check clustering in different length segments

import numpy as np
import pandas as pd
import os
import ipdb
import fnmatch
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA, FastICA
import scipy.stats as stats
from skdata.mnist.views import OfficialImageClassification as OfficialImageClassification
from scipy import signal
import matplotlib.pylab as pylab
import random
from scipy.stats import zscore
from tsne import bh_sne
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


# import scipy.stats as stats
params = {'legend.fontsize': 14,
          'figure.figsize': (12, 10),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':18,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
import matplotlib


'''Usefull functions'''
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
                #if 'base' in filename:
                    #label = '0'
                #elif 'tip' in filename:
                    #label = '1'
                #elif 'seizure' in filename:
                    #label = '2'
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
        data = data_norm    ##data.shape (2048, 1)

    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]

    return data


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


def plotTSNE(data, label,  window=512, num_classes=2, title="t-SNE", save_name='/results'):
    '''tsne clustering on data
    param:
        data: 2d array shape: batch*seq_len*width
        label: 1d array, int labels'''
    from sklearn.manifold import TSNE as TSNE
    tsne = TSNE(n_components=2, perplexity=40.0)
    tsne_results = tsne.fit_transform(data)
    vis_x = tsne_results[:, 0]
    vis_y = tsne_results[:, 1]

    plt.figure()
    plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("cool", num_classes))   ##
    plt.title("t-SNE in orginal {}-long segments".format(window))
    plt.colorbar(ticks=range(num_classes))
    plt.clim(-0.5, num_classes-0.5)
    
    plt.savefig(save_name+"t-SNE-{}.png".format(window), format='png')
    plt.close()

    ### embedding https://medium.com/@pslinge144/representation-learning-cifar-10-23b0d9833c40
    vis_x = (vis_x - np.min(vis_x)) / (np.max(vis_x) - np.min(vis_x))
    vis_y = (vis_y - np.min(vis_y)) / (np.max(vis_y) - np.min(vis_y))
    from PIL import Image
    width = 1000
    height = 800
    max_dim = 25
    full_image = Image.new('RGB', (width, height))
    
    for idx, x in enumerate(data):
        if idx % 100 == 0:
            print(idx, '/', data.shape[0])
        fig, ax = plt.subplots()
        ax.plot(x, 'c')
        fig.patch.set_visible(False)
        ax.axis('off')
        plt.savefig('test.png', format='png')
        plt.close()
        #ipdb.set_trace()
        #x = np.expand_dims(x, 0)
        #tile = Image.fromarray(np.uint8(x * 255).reshape(8, 16))
        tile = Image.open('test.png')
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize((int(tile.width / rs), int(tile.height / rs)),Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim) * vis_x[idx]), int((height-max_dim) * vis_y[idx])))
        
    ipdb.set_trace()
    
        
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
    
    return expand_data.reshape(-1, window, data_x.shape[-1])

def plot_Hierarchy_cluster(data, label, window=512, num_classes=2, title="Hierarchy cluster", save_name='/results'):
    '''apply hierarchy clustering method'''
    data_dist = pdist(data)
    data_link = linkage(data_dist, method='ward')
    dendrogram(data_link, labels=label, leaf_font_size=8, leaf_rotation=0)  ## 
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("distance")
    ipdb.set_trace()
    plt.savefig(save_name+"hierarchy{}.png".format(window), format='png')
    plt.close()


def get_clusters(data, num_clusters=5, window=64, save_name='/results'):
    '''use KMeans cluster to get cluster
    Param:
        data: 2D array, batch*seq_len
        num_clusters: int, num of clusteres that want to apply
        '''
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=num_clusters)
    km.fit(data)
    
    #size = np.ceil(np.sqrt(ind_cluster.size))
    fig, axes = plt.subplots(np.int(np.sqrt(num_clusters)), np.int(np.sqrt(num_clusters)), subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle("cluster on Non-focal signal {} window".format(window), fontsize=22, fontweight='bold')

    for ind, ax in enumerate(axes.flat):
        ind_cluster = np.where(km.labels_ == ind)[0]
          ## figsize=(12, 6),
        #plt.title("Data_F cluster {}".format(cluster))
        #fig.subplots_adjust(hspace=0.3, wspace=0.05)
        #for ii in range(np.minimum(ind_cluster.size, 256)):
        #for ii, ax in enumerate(axes.flat):
            ##ax.plot(data[ind_cluster[ii], :])
        ax.plot(data[ind_cluster, :].T)
                
    
    plt.savefig(save_name+"vis_win_{}_F_samples_from_cluster{}.png".format(window, cluster), format='png')
    plt.close()
                

def get_temporal_corr(data, window=64, stride=64):
    '''get column-wise corr matrix of temporal ordered data segments
    param:
        data: 2D array, seq_len*channel
        '''
    
    #if len(datas.shape) == 2:
    data = np.expand_dims(data, 0)
    data_slide = slide_and_segment(data, window=window, stride=window)
    data = data_slide.T      ##window*num_seq, e.g. 64*128
    df = pd.DataFrame(data)
    corr = df.corr()

    return corr



def plot_average_temporal_corr_F_N(files_F, files_N, window=64, header=None,ifnorm=True):
    ''''''
    num_ave = len(files_F)
    ### Average Focal
    ave_corr_F = 0
    for ind, filename in enumerate(files_F):
        data = read_data(filename, header=0, ifnorm=True)
        
        data_corr = get_temporal_corr(data_slide, window=window, stride=window)
        ave_corr += data_corr
    ave_corr_F = ave_corr_F / num_ave
    
    ## average Non focal
    ave_corr_N = 0
    num_ave = len(files_N)
    for ind, filename in enumerate(files_N):
        data = read_data(filename, header=0, ifnorm=True)
        
        data_corr = get_temporal_corr(data_slide, window=window, stride=window)
        ave_corr_N += data_corr
    ave_corr_N = ave_corr_N / num_ave

    ### plot averaged corr
    '''TODO: feels like average with time doen't make sense!!'''
    vim = -70
    fig, ax = plt.subplots(2, 1)
    ax.flat[0].set_title('Temporal correlation of windowed segments')
    ax.flat[0].matshow(ave_corr_F, interpolation='nearest', aspect='auto')
    ax.flat[0].set_xlabel('time / s')
    ax.flat[0].set_xlim([0, data_v.size/512.0])
    ax.flat[0].set_ylabel('normalized amplitude')
    pxx, freq, t, cax = ax.flat[1].specgram(data_v[:, 0], Fs=512, NFFT=512, cmap='viridis', aspect='auto', vmin=vim)
    ax.flat[1].set_xlabel('time / s')
    ax.flat[1].set_xlim([0, data_v.size/512.0])
    ax.flat[1].set_ylim([0, freq.size])
    ax.flat[1].set_ylabel('frequency / Hz')
    #ipdb.set_trace()
    plt.savefig(save_name+"spectrom_{}_vim{}.png".format(os.path.basename(filename)[0:-4], vim), format='png')
    plt.close()


def plot_single_signal_spectrogram(files, fs=512, save_name='/results'):
    '''plot the spectrogram and signal of individual signal'''
    for ind, filename in enumerate(files):
        data = read_data(filename, header=0, ifnorm=True)
        vim = -70
        ax.flat[0].set_title('Spectrogram')
        ax.flat[0].plot(np.arange(data.size)/512.0, data[:, 0], 'b')
        ax.flat[0].set_xlabel('time / s')
        ax.flat[0].set_xlim([0, data.size/512.0])
        ax.flat[0].set_ylabel('normalized amplitude')
        pxx, freq, t, cax = ax.flat[1].specgram(data[:, 0], Fs=512, NFFT=512, cmap='viridis', aspect='auto', vmin=vim)
        ax.flat[1].set_xlabel('time / s')
        ax.flat[1].set_xlim([0, data.size/512.0])
        ax.flat[1].set_ylim([0, freq.size])
        ax.flat[1].set_ylabel('frequency / Hz')
        #ipdb.set_trace()
        plt.savefig(save_name+"spectrom_{}_vim{}.png".format(os.path.basename(filename)[0:-4], vim), format='png')
        plt.close()

def moving_opt_with_window(values, window=10, stride=5, mode='average'):
    if mode == 'average':
        mask = np.ones(window) / window
        result = np.convolve(values, mask, 'same')
    elif mode == 'sum':
        mask = np.ones(window)
        result = np.convolve(values, mask, 'same')

    return result

def plot_signal_with_average(filename, fs=512, window=256, header=None):
    '''plot signal with its windowed averrage'''
    data = read_data(filename, header=header, ifnorm=True)
    data = data[:, 0]
    data_ave = moving_opt_with_window(data, window=window, mode='average')
    plt.figure()
    plt.title("Origianl signal and window {} average".format(window))
    plt.plot(np.arange(data.size) / 512.0, data, 'c', label='original')
    plt.plot(np.arange(data_ave.size) / 512.0, data_ave, 'm', label='mean')
    plt.xlim([0, data.size/512.0])
    plt.legend(loc='best')
    plt.xlabel("time / s")
    ipdb.set_trace()
    plt.show()
    
def plot_histogram(filename, bins='auto', color='c', header=None):
    '''PLot histogram of data
    for ind, filename in enumerate(files_F[0:15]):
        plot_histogram(files_F[ind], bins=110, color='c', header=None)
    for ind, filename in enumerate(files_N[0:15]):
        plot_histogram(files_N[ind], bins=110, color='m', header=None)
    plt.show()
    '''
    
    data = read_data(filename, header=header, ifnorm=True)
    data = data[:, 0]
    plt.hist(data, bins='auto', color=color, alpha=0.5, label='{}'.format(os.path.basename(filename)[0:6]))
    #plt.legend()
    plt.title("Histogram of {}".format(os.path.basename(filename)[5:14]))
    plt.ylabel("counts")
    plt.xlabel("nomalized values")
    #plt.show()
    

'''Visualization'''


#files = ['data/test_files/oct29_anno_5_pre_seizure.csv', 'data/test_files/oct29_anno_6_pre_seizure.csv']


'''PLot averaged segment temporal correlation
num_ave = 50
windows = np.arange(384, 384, 64)
for ind, window in enumerate(windows):
    fig, axs = plt.subplots(1, 2) ##,, subplot_kw={'xticks': []} 'yticks': []
    plt.title("Average temporal correlation in F and N window{}".format(window), fontsize=22)##, fontweight='bold'
    
    #ipdb.set_trace()
    ave_corr_F = 0
    ave_corr_N = 0
    #for ind, ax in enumerate(axs.flat):
    for jj in range(num_ave):
        
        data_F = read_data(files_F[jj], header=header, ifnorm=True)
        data_N = read_data(files_N[jj], header=header, ifnorm=True)
        data_F = np.expand_dims(data_F, 0)
        data_N = np.expand_dims(data_N, 0)
        ###for window in range(448, 1025, 64):
        
        data_slide_F = slide_and_segment(data_F, window=windows[ind], stride=windows[ind])
        data_slide_N = slide_and_segment(data_N, window=windows[ind], stride=windows[ind])        
        ###Get corr with pandas
        data_F = data_slide_F[:, :, 0].T
        data_N = data_slide_N[:, :, 0].T
        df_F = pd.DataFrame(data_F)
        df_N = pd.DataFrame(data_N)
        ave_corr_F += df_F.corr()
        ave_corr_N += df_N.corr()
    ave_corr_F = ave_corr_F / num_ave
    ave_corr_N = ave_corr_N / num_ave
    #ipdb.set_trace()
    axs.flat[0].imshow(ave_corr_F)
    axs.flat[0].set_title('F window{}'.format(windows[ind]))
    axs.flat[1].set_title('N window{}'.format(windows[ind]))
    axs.flat[1].imshow(ave_corr_N)
    plt.savefig(save_name+"spectrom_F{}_N_win{}_stride{}.png".format(num_ave, windows[ind], window), format='png')
    plt.close()
'''
    #axs.flat[ind+8].imshow(df_N.corr())
    #axs.flat[ind+8].set_title(os.path.basename(files_N[ind])[5:14])

    ####get spectrogram and save as csv'''
    #vmin = -70
    #pxx, freq, t, cax = ax.specgram(data_F[:, 0], Fs=512, cmap='viridis', vmin=vmin)##, vmin=-60
    #ax.set_title(os.path.basename(files_F[ind])[5:14])
    #ax.set_ylim([0, 258])
    #pxx, freq, t, cax = axs.flat[ind+8].specgram(data_N[:, 0], Fs=512, cmap='viridis', vmin=vmin)
    #axs.flat[ind+8].set_ylim([0, 258])
    #axs.flat[ind+8].set_title(os.path.basename(files_N[ind])[5:14])
    #axs.flat[ind+8].pcolormesh(t, f, Sxx_N)
    #axs.flat[ind+8].imshow(Sxx_F, interpolation='nearest', aspect='auto')
    #ipdb.set_trace()
    #plt.savefig(save_name+"spectrom_F{}_N_min{}.png".format(os.path.basename(files_F[ind])[5:14], vmin), format='png')
    #plt.close()
        #get_temporal_corr(data_slide[:, :, 0], window=window, save_name=save_name+os.path.basename(filename)[0:-4])
        #datas[ind, :, :] = data

#plottSNquit()E(datas[:, :, 0], labels, num_classes=3, window=2048, save_name=save_name)

data_dir = 'data/train_data/test_data'   #'data/data_V/segNorm_train_data'  #
save_name = "results/1_ori_data_visual/BBsegments-TSNE/KMeans/F/"
'''slide and seg'''

seq_len = 10240####2048
width = 2
header = None  ##
files_F = find_files(data_dir, pattern='Data_F*.csv', withlabel=False)
files_N = find_files(data_dir, pattern='Data_N*.csv', withlabel=False)
files_all = find_files(data_dir, pattern='Data*.csv', withlabel=True)

num_files = 20
datas = np.zeros((num_files, seq_len, width))
labels = np.array(files_all)[:, 1].astype(np.int)
files_all = np.array(files_all)[:, 0].astype(np.str)

for ind, filename in enumerate(files_all[0:num_files]):
    data = read_data(filename, header=None, ifnorm=True)
    datas[ind, :, :] = data

print labels, np.sum(labels)
    
data_sub = datas
#label_sub = labels[0:500]
for window in range(256, 257, 64):
    num_seg = seq_len // window
    data_slide = slide_and_segment(data_sub, window=window, stride=window)
    #labels_slide = np.repeat(label_sub, num_seg, axis=0)
    #ipdb.set_trace()
    #plotTSNE(data_slide[:, :, 0], labels_slide, num_classes=3, window=window, save_name=save_name)
    get_clusters(data_slide[:, :, 0], num_clusters=50, window=window, save_name=save_name)
    
    #plot_Hierarchy_cluster(data_slide[:, :, 0], labels_slide, num_classes=3, window=window, save_name=save_name)
    print("window {} done".format(window))
