#### util functions for EEG classification
##### file operations
##### data manipulation
##### fploting
import fnmatch
import numpy as np
import tensorflow as tf
import csv
import codecs
from scipy.signal import decimate
import multiprocessing
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
import ipdb
import random
import matplotlib.pylab as pylab
# import scipy.stats as stats
params = {'legend.fontsize': 12,
          'figure.figsize': (10, 8.8),
         'axes.labelsize': 16,
         #'weight' : 'bold',
         'axes.titlesize':16,
         'xtick.labelsize':12,
         'ytick.labelsize':12}
pylab.rcParams.update(params)
import matplotlib

###################### files operation##########################
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
    random.shuffle(files)   # randomly shuffle the files
    return files 

def rename_files(filename):
    os.rename(filename, os.path.dirname(filename) + '/Data_' + os.path.basename(filename)[5:])

def remove_files(filename):
    os.remove(filename)

def multiprocessing_func(data_dir):
    filenames = find_files(data_dir, pattern='*.csv', withlabel=False )
    print filenames
    pool = multiprocessing.Pool()
    version = "save_tfrecord" #'downsampling' #'save_tfrecord'#     #None#'remove'       # 'rename'      # 'rename'        #
    if version == 'downsampling':
        # for ds in [2]:
        pool.map(partial(downsampling, ds_factor=2), filenames)
        print "Downsampling Done!"
    elif version == 'rename':
        pool.map(rename_files, filenames)
        print "rename Done!"
    elif version == 'remove':
        pool.map(remove_files, filenames)
        print "remove Done!"
    elif version == "save_tfrecord":
        pool.map(read_data_save_tfrecord, filenames)
        print "tfrecord saved"
    pool.close()

###################### Data munipulation##########################
def read_data(filename, ifaverage=False, ifnorm=True ):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''
    reader = csv.reader(codecs.open(filename, 'rb', 'utf-8'))
    x_data, y_data = np.array([]), np.array([])
    for ind, row in enumerate(reader):
        x, y = row
        x_data = np.append(x_data, x)
        y_data = np.append(y_data, y)
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(np.float32)
    if ifaverage:
        data = np.expand_dims(np.mean(np.vstack((x_data, y_data)) , axis=0), axis=0 )  ### only get an average signal
    else:
        data = np.vstack((x_data, y_data))  ### read the pair 2 * 10240
    
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        seq_max = np.max(data, axis=1)
        seq_min = np.min(data, axis=1)
        data_norm = np.array([(data[i, :] - seq_min[i]) / (seq_max[i] - seq_min[i]) for i in range(data.shape[0])])

    data = data_norm

    return data.T   #data.T  output data shape (seq_len, channel)

def read_data_save_one_csv(data_dir):
    '''find all files and read each file into one line and save all files' data 'in one  csv.
    Each row is one training sample
    return:
        the first element on each row is the label, then followed by 1*20480 data'''
    #ipdb.set_trace()
    filenames = find_files(data_dir, pattern='Data*.csv', withlabel=False)
    whole_csv = 'data/test_data/test_data.csv'
    #whole_csv = 'data/test_files/test_files.csv'
    whole_data = []
    for ind, filename in enumerate(filenames):
        if 'Data_F' in filename:
            label = 1
        elif 'Data_N' in filename:
            label = 0
        if ind%699 == 0:
            print "ind", ind, "out of ", len(filenames)
        data = read_data(filename, ifaverage=False )   ### falttened data 1 * 20480
        #ipdb.set_trace()
        data = np.hstack((label, data))  #### [label, data1 * 20480]
        whole_data.append(data)
    np.savetxt(whole_csv, np.array(whole_data), header='label, flattened data', delimiter=',', fmt="%10.5f", comments='')


def read_my_data(file_q, num_classes=2):
    '''Read data from queue filenames 
    filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
    '''
    reader = tf.TextLineReader()
    key, value = reader.read(file_q)
    FIELD_DEFAULTS = [[0.0]] * 20481
    value = tf.decode_csv(value, FIELD_DEFAULTS)
    one_hot = tf.one_hot(tf.cast(value[0], tf.int32), num_classes)
    data = value[1:]
    return tf.reshape(data, [2, -1]), one_hot
#filename_queue = tf.train.string_input_producer(["data/whole_train_data/train_data.csv", "data/whole_train_data/test_data.csv"])

def load_train_test_data_queue(data_dir, data_dir_test,  batch_size=20, pattern='Data*.csv', withlabel=True):
    #### Get file names
    #ipdb.set_trace()
    files_train = find_files(data_dir, pattern=pattern, withlabel=False )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0')
    files_test = find_files(data_dir_test, pattern=pattern, withlabel=False)### traverse all the files in the dir, and divide into batches, e.g.  (name, '1'/'0')
    ### convert names to tensor for slicing
    #files_train = tf.convert_to_tensor(files_train, dtype = tf.string)
    #files_test = tf.convert_to_tensor(files_test, dtype = tf.string)
    ### make input file queue
    files_trainq = tf.train.string_input_producer(files_train)
    files_testq = tf.train.string_input_producer(files_test)
    ### preprocessing
    #ipdb.set_trace()
    features_train, labels_train =  read_my_data(files_trainq, num_classes=2)
    features_test, labels_test =  read_my_data(files_testq, num_classes=2)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    ### get shuffled batch
    data_train, labels_train = tf.train.shuffle_batch([features_train, labels_train], batch_size=1, capacity=capacity, min_after_dequeue=min_after_dequeue)
    data_test, labels_test = tf.train.shuffle_batch([features_test, labels_test], batch_size=1, capacity=capacity, min_after_dequeue=min_after_dequeue)

    #with tf.Session() as sess:
        ## Start populating the filename queue.
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)

        #for i in range(1200):
        ## Retrieve a single instance:
        #example, label = sess.run([features, col5])

    #coord.request_stop()
    #coord.join(threads)

    return data_train, labels_train, data_test, labels_test

def input_parser(row, num_classes=2):
    '''tensorflow map this function to all the files. read the whole file as a training sample'''
    FIELD_DEFAULTS = [[0.0]] * 20481
    ipdb.set_trace()
    value = tf.decode_csv(row, FIELD_DEFAULTS)
    ### convert the label to a one-hot encoding
    one_hot = tf.one_hot(value[0], num_classes)
    #### read the file
    data = value[1:]
    return data, one_hot
    
def load_train_test_data(data_dir, data_dir_test,  batch_size=20, pattern='Data_*.csv', withlabel=True):
    '''get filenames in data_dir, and data_dir_test, put them into dataset'''
    with tf.name_scope("Data"):
        #### Get file names
        files_train = find_files(data_dir, pattern=pattern, withlabel=False )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0') or (name)
        files_test = find_files(data_dir_test, pattern=pattern, withlabel=False)### traverse all the files in the dir, and divide into batches, e.g.  (name, '1'/'0') or (name)
        # create TensorFlow Dataset objects
        dataset_train = tf.data.Dataset.from_tensor_slices(files_train)
        dataset_test = tf.data.Dataset.from_tensor_slices(files_test)
        ### map self-defined functions to the dataset
        ipdb.set_trace()
        dataset_train = dataset_train.flat_map(lambda filename: tf.data.TextLineDataset(filename).map(input_parser)).repeat().batch(batch_size).shuffle(buffer_size=256)
        dataset_test = dataset_test.flat_map(lambda filename: tf.data.TextLineDataset(filename).map(input_parser)).repeat().batch(batch_size).shuffle(buffer_size=10000)

        # create TensorFlow Iterator object
        iter = dataset_train.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
        ele = iter.get_next()   #you get the filename
        ele_test = iter_test.get_next()   #you get the filename
        return ele, ele_test, iter, iter_test

def load_files_train_test_data(data_dir, data_dir_test):
    files_train = find_files(data_dir, pattern=pattern, withlabel=False )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0') or (name)
    files_test = find_files(data_dir_test, pattern=pattern, withlabel=False)
        

def load_and_save_data(data_dir, data_dir_test, pattern='Data*.csv', withlabel=True, ifaverage=False, num_classes=2):
    '''Keras way of loading data
    return: x_train, y_train, x_test, y_test'''
    #### Get data
    files_train = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
    files_test = find_files(data_dir_test, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, (name, '1'/'0')
    data_train = []
    labels_train = np.empty([0])
    for ind in range(len(files_train)):
        if ind % 100 == 0:
            print "train", ind
        #data = read_data(files_train[ind][0], ifaverage=ifaverage)
        #data_train.append(data)
        labels_train = np.append(labels_train, files_train[ind][1])
    #labels_train =  np.eye((num_classes))[labels_train.astype(int)]   # get one-hot lable

    data_test = []
    labels_test = np.empty([0])
    for ind in range(len(files_test)):
        if ind % 100 == 0:
            print "test", ind
        #data = read_data(files_test[ind][0], ifaverage=ifaverage)
        #data_test.append(data)
        labels_test = np.append(labels_test, files_test[ind][1])
    #labels_test =  np.eye((num_classes))[labels_test.astype(int)]   # get one-hot lable
    np.savetxt(save_name, data, header=header, delimiter=',', fmt="%10.5f", comments='')

    np.savez("testF751testN1501_ori", x_train=np.array(data_train), y_train=np.array(labels_train), x_test=np.array(data_test), y_test=np.array(labels_test))
    return np.array(data_train), np.array(labels_train), np.array(data_test), np.array(labels_test)

#x_train=xx_train, y_train=yy_train, x_test=xx_test, y_test=yy_test


def load_my_data():
    '''given the data dir, load train and test'''

def my_input_fn(file_path, perform_shuffle=False, epochs=1, num_classes=2):
    def _parse_line(line):
        ### decode the line into variables
        record_defaults = [[1.]] * 20481
        data = tf.decode_csv(line, record_defaults=record_defaults)   # each line with two values no headere
        label, data = data[0], data[1:]
        return label, data

    dataset = (tf.data.TextLineDataset(file_path) # Read text file
                    .skip(1) # Skip header row
                    .map(decode_csv)) # Transform each elem by applying decode_csv fn
    if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(epochs) # Repeats dataset this # times
    dataset = dataset.batch(32)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def downsampling(filename, ds_factor):
    """Downsample the original data by the factor of ds_factor to get a low resolution version"""
    x, y = read_data(filename)
    ds_x =  decimate(x, ds_factor)
    ds_y =  decimate(y, ds_factor)
    np.savetxt(os.path.dirname(filename) + "/ds_" + np.str(ds_factor)  + os.path.basename(filename) , zip(ds_x, ds_y), delimiter=',', fmt="%10.5f")

def save_data(data, header='data', save_name="save_data"):
    '''save data into a .csv file
    data: list of data that need to be saved, (x1, x2, x3...)
    header: String that will be written at the beginning of the file.'''
    np.savetxt(save_name, data, header=header, delimiter=',', fmt="%10.5f", comments='')

def load_data(data_dir):
    '''Load variables' data from pre-saved .csv file
    return a dict '''
    reader = csv.reader(codecs.open(data_dir, 'rb', 'utf-8'))
    data = dict()
    for ind, row in enumerate(reader):
        if  ind == 0:
            names = row
        else:
            data[names[ind-1]]= np.array(row).astype(np.float32)
    return data, names

def smooth(x, window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    """
    #if x.ndim != 1:
        #raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def opp_sliding_window(data_x, data_y, ws, ss):
    '''apply a sliding window to original data and get segments of a certain window lenght
    e.g.
    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    '''
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def sliding_window(data_x, data_y, window=128, stride=64):
    '''
    Param:
        datax: array-like data shape (num_seq, seq_len, channel)
        data_y: shape (num_seq, num_classes)
        window: int, number of frames to stack together to predict future
        noverlap: int, how many frames overlap with last window
    Return:
        expand_x : shape(num_seq, num_segment, window)
        expand_y : shape(num_seq, num_segment, num_classes)
        '''
    num_seg = (data_x.shape[1] - window + 1) // stride
    shape = (data_x.shape[0], num_seg, window)      ## done change the num_seq
    strides = (data_x.itemsize, data_x.itemsize * stride, data_x.itemsize)    ##
    expand_x = np.lib.stride_tricks.as_strided(data_x, shape=shape, strides=strides).reshape(-1, window)
    expand_y = np.repeat(data_y,  num_seg, axis=0).reshape(data_y.shape[0], num_seg, data_y.shape[1]).reshape(-1, data_y.shape[1])
    return expand_x, expand_y

#def normalize_data(data):

###################### plots ##########################
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

def plot_learning_curve(train_scores, test_scores , num_trial=1, title="Learning curve", save_name="learning curve"):
    '''plot smooth learning curve
    train_scores: n_samples * n_features
    test_scores: n_samples * n_features
    '''
    plt.figure()
    plt.title(title)
    plt.xlabel("training batches")
    plt.ylabel("accuracy")
    if num_trial > 1:
        train_sizes = len(train_scores)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(np.arange(train_sizes), train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.25,color="m")
        plt.fill_between(np.arange(train_sizes), test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.25, color="c")
        plt.plot(np.arange(train_sizes), train_scores_mean, '-', color="indigo",
                 label="Training score")
        plt.plot(np.arange(train_sizes), test_scores_mean, '-', color="teal",
                 label="Test score")
        plt.legend(loc="best")
    elif num_trial == 1:
        train_scores_mean = smooth(train_scores_mean, window=11)
        data_smooth = train_scores_mean.shape[0]
        sizes = data.shape[0]
        plt.grid()
        plt.fill_between(np.arange(train_scores), data_smooth - data_std, data_smooth + data_std, alpha=0.3, color="lightskyblue")
        plt.plot(np.arange(sizes), data_smooth, '-', color="royalblue")
    plt.savefig(save_name, format="png")
    plt.close()

def plot_smooth_shadow_curve(datas, colors=['darkcyan'], xlabel='training batches / 20', ylabel='accuracy', title='Loss during training', labels='accuracy_train', save_name="loss"):
    '''plot a smooth version of noisy data with mean and std as shadow
    data: a list of variables values, shape: (batches, )
    color: list of prefered colors
    '''
    plt.figure()
    fill_colors = ['lightcoral', 'plum']
    for ind, data in enumerate(datas) :
        data_smooth = smooth(data, window_len=25)
        data_smooth = data_smooth[0:len(data)]
        data_mean = np.mean(np.vstack((data, data_smooth)), axis=0)
        data_std = np.std(np.vstack((data, data_smooth)), axis=0)
        sizes = data_std.shape[0]
        plt.grid()
        plt.fill_between(np.arange(sizes), data_smooth - data_std, data_smooth + data_std, alpha=0.5, color=fill_colors[ind])
        plt.plot(np.arange(sizes), data_smooth, '-', linewidth=2, color=colors[ind], label=labels[ind])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="best")
    plt.title(title)
    plt.savefig(save_name, format="png")
    plt.close()

def plotdata(data, color='darkorchid', xlabel="training time", ylabel="loss", save_name="save"):
    '''
    data: 1D array '''
    plt.figure()
    plt.plot(data, color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylabel == 'accuracy':
        plt.ylim([0.0, 1.05])
    elif ylabel == 'loss':
        plt.ylim([0.0, 1.0])
    plt.savefig(save_name + "_{}".format(ylabel))
    plt.close()
###################### Data munipulation##########################


if __name__ == "__main__":
    #data_dir = "data/train_data"
    data_dir_test = "data/test_data"
    #data_dir = 'data/test_files'
    # read_data_save_tfrecord(data_dir)
    #for ind, dirr  in enumerate(data_dir):
        #multiprocessing_func(dirr )
    # get_Data(data_dir, data_dir_test, pattern='Data*.csv', withlabel=True)
    # multiprocessing_func(data_dir)
    # read_from_tfrecord("data/test_files/test_files.tfrecords")
    #read_tfrecord()
    read_data_save_one_csv(data_dir_test)
    #filename = "data/train_data/train_data.csv"
    #read_data(filename)










###
#def load_train_test_data_queue(data_dir, data_dir_test,  batch_size=20, pattern='Data*.csv', withlabel=True):
    ##### Get file names
    #files_wlabel_train = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0')
    #files_wlabel_test = find_files(data_dir_test, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, e.g.  (name, '1'/'0')
    #files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
    #files_test, labels_test = np.array(files_wlabel_test)[:, 0], np.array(files_wlabel_test)[:, 1].astype(np.int)   ##
    #### convert names to tensor for slicing
    #files_train = tf.convert_to_tensor(files_train, dtype = tf.string)
    #files_test = tf.convert_to_tensor(files_test, dtype = tf.string)
    #### make input file queue
    #files_trainq = tf.train.string_input_producer(files_train)
    #files_testq = tf.train.string_input_producer(files_test)
    #### preprocessing
    #features_train = read_my_file_format(files_trainq)
    #features_test = read_my_file_format(files_testq)

    #min_after_dequeue = 10000
    #capacity = min_after_dequeue + 3 * batch_size
    #### get shuffled batch
    #data_train, labels_train = tf.train.shuffle_batch([features_train, labels_train], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    #data_test, labels_test = tf.train.shuffle_batch([features_test, labels_test], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

    #return data_train, labels_train, data_test, labels_test


#def load_train_test_data(data_dir, data_dir_test,  batch_size=20, pattern='Data*.csv', withlabel=True):
    #'''get filenames in data_dir, and data_dir_test, put them into dataset'''
    #with tf.name_scope("Data"):
        ##### Get file names
        #files_wlabel_train = find_files(data_dir, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0')
        #files_wlabel_test = find_files(data_dir_test, pattern=pattern, withlabel=withlabel )### traverse all the files in the dir, and divide into batches, e.g.  (name, '1'/'0')

        #files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
        #files_test, labels_test = np.array(files_wlabel_test)[:, 0], np.array(files_wlabel_test)[:, 1].astype(np.int)   ##         seperate the name and label
        ## create TensorFlow Dataset objects
        #dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        #dataset_test = tf.data.Dataset.from_tensor_slices((files_test, labels_test)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        #### map self-defined functions to the dataset
        #dataset_train = dataset_train.map(input_parser)
        #dataset_test = dataset_test.map(input_parser)
        ## create TensorFlow Iterator object
        #iter = dataset_train.make_initializable_iterator()
        #iter_test = dataset_test.make_initializable_iterator()
        #ele = iter.get_next()   #you get the filename
        #ele_test = iter_test.get_next()   #you get the filename
        #return ele, ele_test, iter, iter_test

########### multiprocessing read files and save to one .csv ###############3
#def Writer(dest_filename, some_queue, some_stop_token):
    #with open(dest_filename, 'w') as dest_file:
        #while True:
            #line = some_queue.get()
            #if line == some_stop_token:
                #return
            #dest_file.write(line)

#def the_job(some_queue):
    #for item in something:
        #result = process(item)
        #some_queue.put(result)

#def multiprocessing_save_csv(data_dir):
    #'''Deploy reading-file work to pool, and collect the results and write them in ONE .csv file'''
    ##pool = multiprocessing.Pool()
    ##with open('data/test_files/test_files.csv') as source:
        ##results = pool.map()
    #filenames = find_files(data_dir, pattern='Data*.csv', withlabel=False)
    #queue = multiprocessing.Queue()
    #STOP_TOKEN="STOP!!!"
    #writer_process = multiprocessing.Process(target = Writer, args=( 'data/test_files/test_files.csv', queue, STOP_TOKEN))
    #writer_process.start()

    ## Dispatch all the jobs

    ## Make sure the jobs are finished

    #queue.put(STOP_TOKEN)
    #writer_process.join()
    ## There, your file was written.

   
#def read_data_save_tfrecord(data_dir):
    #'''find all files and save them into a .tfrecord file. Each file is an entry of .tfrecord'''
    #filenames = find_files(data_dir, pattern='*.csv', withlabel=False)
    #tfrecord_file =  'data/test_files/test_files.tfrecords'
    #writer = tf.python_io.TFRecordWriter(tfrecord_file)
    #for ind, filename in enumerate(filenames):
        #reader = csv.reader(codecs.open(filename, 'rb', 'utf-8'))
        #if 'F_' in filename:
            #label = 1
        #elif 'N_' in filename:
            #label = 0
        #example = tf.train.Example()
        #for ind, row in enumerate(reader):
            #row = np.array(row).astype(np.float32)
            #if ind%10000 == 0:
                #print "file:", filename, "ind: ", ind, row

            #example.features.feature['features'].float_list.value.extend(row)
        ## shape = np.array([ind, 2])
        #example.features.feature['label'].int64_list.value.append(label)
        ## example.features.feature['shape'].float_list.value.extend(shape)
        #writer.write(example.SerializeToString())
    #writer.close()

#def read_from_tfrecord(filename):
    #'''read tfrecord'''
    #tfrecord_file_queue = tf.train.string_input_producer(filename, name='queue')
    #reader = tf.TFRecordReader()
    #_, tfrecord_serialized = reader.read(tfrecord_file_queue)

    #tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                #features={
                    #'label': tf.FixedLenFeature([], tf.string),
                    #'features': tf.FixedLenFeature([], tf.string)}, name="tf_features")
    #features = tf.decode_raw(tfrecord_features['features'], tf.float32)
    #label = tf.decode_raw(tfrecord_features['label'], tf.int)
    #print features.shape, label

#def read_tfrecord():
    #data_path = "data/test_files/test_files.tfrecords"

    #with tf.Session() as sess:
        #feature = {'data': tf.FixedLenFeature([], tf.string),
                    #'label': tf.FixedLenFeature([], tf.int64)}
        #### Create a list of filenames and pass it to a queue
        #filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

        ## Define a reader and read the next record
        #reader = tf.TFRecordReader()
        #_, serialized_example = reader.read(filename_queue)

        ## Decode the record read by the reader
        #features = tf.parse_single_example(serialized_example, features=feature)

        ## Convert the image data from string back to the numbers
        #data = tf.decode_raw(features['data'], tf.float32)
        #label = tf.cast(features['label'], tf.int32)

        ###Creates batches by randomly shuffling tensors
        #datas, labels = tf.train.shuffle_batch([data, label], batch_size=3, capacity=30, num_threads=1, min_after_dequeue=10)
        ## Initialize all global and local variables
        #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        #sess.run(init_op)

        ## Create a coordinator and run all QueueRunner objects
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)

        #for batch in range(3):
            #data, label = sess.run([datas, labels])
            #print data.shape, label

        ## Stop the threads
        #coord.request_stop()
        ## Wait for threads to stop
        #coord.join(thread)

