import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pandas
import math
# from keras.datasets import mnist
import ipdb
import keras
import os
import matplotlib.pyplot as plt
import functions as func
# import fnmatch
# import random
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pylab as pylab
params = {'legend.fontsize': 12,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)

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

def plotTSNE(data, labels, num_classes=2, n_components=3, title="t-SNE", save_name='/results', postfix='band_PSD'):
    '''tsne clustering on data
    param:
        data: 2d array shape: batch*seq_len*width
        label: 1d array, int labels'''

    from tsne import bh_sne
    tsne_results = bh_sne(data, d=n_components)
    #tsne_results = TSNE(n_components=3, random_state=99).fit_transform(data)
    
    #colors =plt.cm.get_cmap("cool", num_classes)
    colors = ['c', 'm']
    target_names = ['non_focal', 'focal']
    fig = plt.figure()
    if n_components == 3:
        vis_x = tsne_results[:, 0]
        vis_y = tsne_results[:, 1]
        vis_z = tsne_results[:, 2]
        ax = fig.add_subplot(111, projection='3d')
        
        for i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], tsne_results[labels == i, 2], color=color, alpha=.8,label=target_name)
    elif n_components == 2:
        vis_x = tsne_results[:, 0]
        vis_y = tsne_results[:, 1]
        ax = fig.add_subplot(111)
        for color, i, target_name in zip(colors, np.arange(num_classes), target_names):
            ax.scatter(tsne_results[labels == i, 0], tsne_results[labels == i, 1], color=color, alpha=.8, label=target_name)###lw=2,
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    #plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("cool", num_classes))   ##
    plt.title("t-SNE-{}".format(postfix))
    plt.savefig(save_name+"t-SNE-{}.png".format(postfix), format='png')
    plt.close()


batch_size = 64
epochs = 20
'''MNIST'''
#batch_size = 128
#num_classes = 10
#epochs = 10

## input image dimensions
#img_x, img_y = 28, 28

## load the MNIST data set, which already splits into train and test sets for us
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

## reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
## because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
#x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
#x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
#input_shape = (img_x, img_y, 1)

## convert the data to the right type
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print('x_train shape:', x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

## convert class vectors to binary class matrices - this is for use in the
## categorical_crossentropy loss below
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)

'''BB dataset'''
# nData = 1500
# num_classes = 2
# xData = np.ndarray(shape=(nData,3,1))
#
# data_dir = 'data/data'
# files_wlabel = find_files(data_dir, pattern='Data*.csv', withlabel=True)
# files, yData = np.array(files_wlabel)[:, 0].astype(np.str), np.array(np.array(files_wlabel)[:, 1]).astype(np.int)
# for ind, filename in enumerate(files):
#     if ind % 200 == 0:
#         print(filename)
#     data = pandas.read_csv(filename, header=None)
#     #print(data)
#     xData[ind,:,:]= data.values
# xData = zscore(xData)

# print(xData.shape)

'''BB entropy features'''
## load focal features, nonfocal features, shuffle
#data_dir = 'entropy'
#save_name = "results/"
#num_files = 3750 * 2
#num_classes = 2
#batch_size = 300
#entropies = np.ones((num_files, 6)) ### label, ae_entropy, sp_entropy, re_entropy
#labels = np.ones((num_files))
#entropies[0:3750, 0:2] = pd.read_csv(data_dir+'/ae-focal.csv', header=None)
#entropies[0:3750, 2:4] = pd.read_csv(data_dir+'/se-focal.csv', header=None)
#entropies[0:3750, 4:6] = pd.read_csv(data_dir+'/re-focal.csv', header=None)

### non focal
#entropies[3750:, 0:2] = pd.read_csv(data_dir+'/ae-nonfocal.csv', header=None)
#entropies[3750:, 2:4] = pd.read_csv(data_dir+'/se-nonfocal.csv', header=None)
#entropies[3750:, 4:6] = pd.read_csv(data_dir+'/re-nonfocal.csv', header=None)
#labels[3750:] = 0

'''PSD'''
num_classes = 2
data_dir = 'data/PSD/'
save_name = "results/"
feature_width = 12
data_train = pd.read_csv(data_dir+'band_PSD_train.csv', header=0).values
y_train, x_train = data_train[:, 0].astype(np.int), data_train[:, 1:]

data_test = pd.read_csv(data_dir+'band_PSD_test.csv', header=0).values
y_test, x_test = data_test[:, 0].astype(np.int), data_test[:, 1:]

y_train_hot = keras.utils.to_categorical(y_train, num_classes)
y_test_hot = keras.utils.to_categorical(y_test, num_classes)
print("Done, x_train.shape", x_train.shape)

#ipdb.set_trace()
#plotTSNE(x_train, y_train, num_classes=num_classes, n_components=2, save_name='./')

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
##ipdb.set_trace()
#ax.scatter(x_train[:, 0], x_train[:, 2],  x_train[:, 4], c=y_train, cmap=plt.cm.get_cmap("cool", num_classes), alpha=0.5)
#plt.legend()
#ax.set_xlabel("Approximate entropy")
#ax.set_ylabel("Sample entropy")
#ax.set_zlabel("Reyni entropy")
#ax.set_title('3D scatter of original entropy data')

#plt.colorbar()
#ax.scatter(entropies[3750:, 0], entropies[3750:, 2],  entropies[3750:, 4], c='c', marker='*')

### plot entropy bar chart
#mean_focal = np.mean(entropies[0:3750, :], axis=0)
#mean_nonfocal = np.mean(entropies[3750:, :], axis=0)
#std_focal = np.std(entropies[0:3750, :], axis=0)
#std_nonfocal = np.std(entropies[3750:, :], axis=0)

#print("mean_focal", mean_focal, "mean_nonfocal", mean_nonfocal)
#barWidth = 0.3
#r1 = np.arange(6)
#r2 = [x + barWidth for x in r1]

#plt.figure()
#plt.bar(r1, mean_focal, yerr = std_focal, color='c', width=barWidth, edgecolor='white', label='focal')
#plt.bar(r2, mean_nonfocal, yerr=std_nonfocal, color='m', width=barWidth, edgecolor='white', label='non-focal')
#plt.legend()
#plt.title("Entropy statistics in focal and non-focal")
#plt.xticks([r + barWidth for r in range(len(mean_focal))], ['ae_1', 'ae_2','se_1', 'se_2', 're_1', 're_2'])
#plt.savefig('results/Entropies.png', format='png')
#plt.close()

## split training and testing
#ipdb.set_trace()
'''cnn Data reshape'''
# xData = np.expand_dims(xData, 3)
# nTrain = math.ceil(nData*0.7)
# x_train = xData[0:nTrain,:,:, :]
# y_train = np.array(yData[0:nTrain])
# x_test = xData[nTrain:,:,:, :]   ## Me: xData[nTrain:,:,:]    :P

'''FNN'''
#nTrain = math.ceil(nData*0.7)
#x_train = xData[0:nTrain,:,:]
#y_train = np.array(yData[0:nTrain])
#x_test = xData[nTrain:,:,:]   ## Me: xData[nTrain:,:,:]    :P


# y_test = np.array(yData[nTrain:])
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#print(yData[0:20])
''' FNN MODEL'''
model = Sequential()
# model.add(Flatten())
#model.add(Dense(500, input_shape=(10239,2), activation='relu'))
model.add(Dense(100, input_shape=(feature_width,), activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(300, activation='relu'))
# model.add(Dropout(0.75))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

'''CNN MODEL'''
# model = Sequential()
# model.add(Conv2D(4, kernel_size=(2, 2),
#                  activation='relu',
#                  input_shape=(6,1)))
# model.add(BatchNormalization())
# #model.add(MaxPooling2D(pool_size=(2, 1), strides=(2, 1)))
# model.add(Conv2D(8, (2, 2), strides=(1, 1), activation='relu'))
# #model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(BatchNormalization())
# model.add(Conv2D(16, (2, 2), strides=(1, 1), activation='relu'))
# #model.add(MaxPooling2D(pool_size=(2, 1)))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(50, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

'''ResNet'''

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 #activation='relu',
                 #input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr=0.0005),
              metrics=['accuracy'])

model.fit(x_train, y_train_hot,
          epochs=3000,
          shuffle = True, 
          batch_size=batch_size,
          validation_data = (x_test, y_test_hot))

print("testing:\n")
score = model.evaluate(x_test, y_test_hot, batch_size=128)
print('final score', score)
