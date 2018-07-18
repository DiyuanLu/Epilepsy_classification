#### THis is a script to load model and contimue training/ plotting
import numpy as np
import ipdb
import sys
import functions as func
import tensorflow as tf



def save_data_to_TFRecord(data_dir, num_per_file=500, pattern='Data*.csv', prefix='train'):
    '''read data from data_dir and save them into TFReocrds
    Param:
        data_dir: the dir of all the data
        num_per_file: how many data to group together save as one tfrecord
        '''
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # Let's collect the real images to later on compare
    # to the reconstructed ones
    original_images = []

    
    
    files_wlabel = func.find_files(data_dir, pattern=pattern, withlabel=True)### traverse all the

    files, labels = np.array(files_wlabel)[:, 0].astype(np.str), np.array(np.array(files_wlabel)[:, 1]).astype(np.int)

    for num in range(len(files) // num_per_file):   ### number of files to  write into
        batch_files = files[num*num_per_file : (num+1)*num_per_file]

        tfrecords_filename = data_dir + prefix+ '-Data_norm_batch{}.tfrecords'.format(num)
        writer = tf.python_io.TFRecordWriter(tfrecords_filename)
        
        for ind, filename in enumerate(batch_files):

            data = func.read_data(filename,  header=None, ifnorm=True)
            seq_len = data.shape[0]
            width = data.shape[1]

            original_images.append((data, labels[ind]))

            label = labels[ind]

            # Create a feature
            feature = {'label': _int64_feature(label),
                   'data': _bytes_feature(tf.compat.as_bytes(data.tostring()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        writer.close()
    sys.stdout.flush()


data_dir = 'data/Whole_data/validate_data/'
save_data_to_TFRecord(data_dir, num_per_file=500, pattern='Data*.csv', prefix='val')
#files = func.find_files(data_dir, pattern='*.tfrecords', withlabel=False)

'''Load the tfrecords '''
with tf.Session() as sess:
    feature = {'data': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64)}
     Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer(files, num_epochs=10)   ### the files have to a list

     Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

     Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
     Convert the image data from string back to the numbers
    data = tf.decode_raw(features['data'], tf.float64)
    
     Cast label data into int32
    labels = tf.cast(features['label'], tf.int32)   ### the feature name should be exactly the same as you save them
    
    ## define the shape
    data = tf.reshape(data, [10240, 2, 1])
     Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([data, labels], batch_size=15, capacity=50000, num_threads=1, min_after_dequeue=10000)

     Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
     Create a coordinator and run all QueueRunner objects
    ipdb.set_trace()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for batch_index in range(10):
        img, lbl = sess.run([images, labels])
        print('label: ', lbl, 'data shape: ', img.shape)
        for j in range(6):
            plt.subplot(2, 3, j+1)
            plt.plot(img[j,:,0])
            #plt.title('cat' if lbl[j]==0 else 'dog')
        plt.show()
     Stop the threads
    coord.request_stop()
    
     Wait for threads to stop
    coord.join(threads)
    sess.close()


    
