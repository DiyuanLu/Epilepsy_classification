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
import ipdb

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

def multiprocessing_func():
    data_dir = "data/sub_test"
    files_train = find_files(data_dir, withlabel=False )
    pool = multiprocessing.Pool()
    for ds in [16]:
        pool.map(partial(downsampling, ds_factor=ds), files_train)
        print "Done!"
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
                    train_scores_mean + train_scores_std, alpha=0.3,color="m")
    plt.fill_between(np.arange(train_sizes), test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.4, color="c")
    plt.plot(np.arange(train_sizes), train_scores_mean, '-', color="b",
             label="Training score")
    plt.plot(np.arange(train_sizes), test_scores_mean, '-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(save_name, format="png")
    plt.close()

def plot_smooth_shadow_curve(data, title="Loss during training", save_name="loss"):
    '''plot a smooth version of noisy data with mean and std as shadow
    data: n_samples * n_trials'''
    data_mean = np.mean(data, axis=1)
    data_std = np.std(data, axis=1)
    sizes = data.shape[0]
    plt.grid()
    plt.fill_between(np.arange(sizes), data_mean - data_std, data_mean + data_std, alpha=0.5, color="c")
    plt.plot(np.arange(sizes), data_mean, '-', color="b")
    plt.ylim([0.0, 10])
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
        plt.ylim([0.0, 1.2])
    elif ylabel == 'loss':
        plt.ylim([0.0, 10])
    plt.savefig(save_name + "_{}".format(ylabel))
    plt.close()
