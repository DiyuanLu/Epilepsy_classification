

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

#def load_train_test_data_queue(data_dir, data_dir_test,  batch_size=20, pattern='Data*.csv', withlabel=True):
    ##### Get file names
    ##ipdb.set_trace()
    #files_train = find_files(data_dir, pattern=pattern, withlabel=False )### traverse all the files in the dir, and divide into batches, e.g. (name, '1'/'0')
    #files_test = find_files(data_dir_test, pattern=pattern, withlabel=False)### traverse all the files in the dir, and divide into batches, e.g.  (name, '1'/'0')
    #### convert names to tensor for slicing
    ##files_train = tf.convert_to_tensor(files_train, dtype = tf.string)
    ##files_test = tf.convert_to_tensor(files_test, dtype = tf.string)
    #### make input file queue
    #files_trainq = tf.train.string_input_producer(files_train)
    #files_testq = tf.train.string_input_producer(files_test)
    #### preprocessing
    ##ipdb.set_trace()
    #features_train, labels_train =  read_my_data(files_trainq, num_classes=2)
    #features_test, labels_test =  read_my_data(files_testq, num_classes=2)

    #min_after_dequeue = 10000
    #capacity = min_after_dequeue + 3 * batch_size
    #### get shuffled batch
    #data_train, labels_train = tf.train.shuffle_batch([features_train, labels_train], batch_size=1, capacity=capacity, min_after_dequeue=min_after_dequeue)
    #data_test, labels_test = tf.train.shuffle_batch([features_test, labels_test], batch_size=1, capacity=capacity, min_after_dequeue=min_after_dequeue)

    #return data_train, labels_train, data_test, labels_test
