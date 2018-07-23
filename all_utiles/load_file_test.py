import tensorflow as tf
import functions as func
import ipdb

#data_dir = "data/test_files/"
##data_dir_test = "data/test_data"
#data_dir_test = "data/test_files/"

#ele, ele_test, iter, iter_test =  func.load_train_test_data(data_dir, data_dir_test,  batch_size=3, pattern='Data_*.csv', withlabel=False)
def decode_csv(line):
    record_defaults = [[0.0], [0.0]]
    parsed_line = tf.decode_csv(line, record_defaults)
    #label = parsed_line[0]          #NOT needed. Only if more than 1 column makes the label...
    #del parsed_line[0]
    features = parsed_line[1:]  # Stack features so that you can later vectorize forward prop., etc.
    return_value = features, label
    return return_value

def read_whole_csv(filename):
    with open(filename, 'r') as f:
        for ind, line in enumerate(f.readlines()) :
            record = line.rstrip().split(', ')
            #ipdb.set_trace()
            if ind%10000 == 0:
                print ind
            features = [tf.cast(n, tf.float32) for n in record]
            features = tf.stack(features)
            #ipdb.set_trace()
        yield features
        #return features\

#step 1: Convert to csv data to tfrecords data. Example code below (see read_csv from from_generator example above).
with tf.python_io.TFRecordWriter("my_train_dataset.tfrecords") as writer:
    for features, labels in read_csv('my_train_dataset.csv'):
        example = tf.train.Example()
        example.features.feature[
            "features"].float_list.value.extend(features)
        example.features.feature[
            "label"].int64_list.value.append(label)
        writer.write(example.SerializeToString())
# Step 2: Write a dataset the decodes these record files.
def parse_function(example_proto):
    features = {
        'features': tf.FixedLenFeature((n_features,), tf.float32),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features['features'], parsed_features['label']


def get_dataset():
    dataset = tf.data.TFRecordDataset(['data/test_files/test_files.tfrecords'])
    dataset = dataset.map(parse_function)
    return dataset

def get_inputs(batch_size, shuffle_size):
    dataset = get_dataset()  # one of the above implementations
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.repeat()  # repeat indefinitely
    dataset = dataset.batch(batch_size)
    features, label = dataset.make_one_shot_iterator().get_next()       
    
            
            
#ipdb.set_trace()
#filenames = ['data/whole_train_data/test_data/test_data.csv']
#filenames = ['data/test_files/Data_F_Ind_3009.csv', 'data/test_files/Data_F_Ind_3010.csv', 'data/test_files/Data_F_Ind_3011.csv', 'data/test_files/Data_F_Ind_3012.csv
### working version reading line by line
#dataset5 = tf.data.Dataset.from_tensor_slices(filenames)
#dataset5 = dataset5.flat_map(lambda filename: tf.data.TextLineDataset(filename).skip(1).map(decode_csv))
### generator
#generator = lambda: read_whole_csv(filenames)
#dataset5 = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))

#dataset5 = dataset5.map(read_whole_csv)
#dataset5 = dataset5.map(decode_whole_csv)     ##Maps map_func across this dataset and flattens the result

## 
#dataset5 = dataset5.batch(4).shuffle(buffer_size=10000).repeat(20)   ###repeat().

#iterator5 = dataset5.make_initializable_iterator()
#next_element5 = iterator5.get_next()


with tf.Session() as sess:
    # Train 2 epochs. Then validate train set. Then validate dev set.
    sess.run(iterator5.initializer)
    for _ in range(10):     
        features, label = sess.run(next_element5)
              # Train...
        #print("shape:", features.shape)
        print("label", label, features.shape)

    # Validate (cost, accuracy) on train set
    ipdb.set_trace()
    print("\nDone with the first iterator\n")
        
