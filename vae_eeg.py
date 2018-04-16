

#### This is a script treat the eeg data(2250 clips) as sequence data and apply VAE to encode feateures
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
#matplotlib.use('Agg')
import os
from scipy.misc import imsave
#import data
import ipdb
import datetime
import functions as func
#from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
#mnist = input_data.read_data_sets("data/MNIST_data/",  one_hot=True)

version = "vae_CNN_eeg"     ###"VAE_ds16"  #
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
data_dir = "data/train_data"
data_dir_test = "data/test_data"
save_every = 25
plot_every = 20
test_every = 5
num_iterations = 101   # 50
recording_interval = 1000    # 1000   #
print_result = 10
### Hyperparams
seq_len = 1280 #   640     #10240    #28 * 28
hid_dim1 = 2000   # Encoder: input -- hidden1 -- latent1 -- hidden2 -- latent2
hid_dim2 = 500
latent_dim = 50
batch_size = 16
epochs = 50
total_batches =  epochs * 3000 // batch_size + 1
results_dir = "results/" + version + '/' + datetime +'bs_' +np.str(batch_size)+'len_' +np.str(seq_len)
logdir = results_dir  + "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir ):
    os.makedirs(results_dir )
    
def get_test_data():
    test_data = np.empty([0, seq_len])
    test_labels = np.empty([0])
    for filen in files_test:
        data = np.average(func.read_data(filen[0]), axis=0)
        test_data = np.vstack((test_data, data))
        test_labels = np.append(test_labels, filen[1])
    test_labels = np.eye((n_classes))[test_labels.astype(int)]
    return test_data, test_labels


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def FC_Layer(X, W, b):
    return tf.matmul(X, W) + b

def upsample(inputs, name='depool', factor=[1,2]):
    size = [int(inputs.shape[1] * factor[0]), int(inputs.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(inputs, size=size, align_corners=None, name=None)
    return out

#### input
with tf.name_scope("input"):
    ## the real data from database
    inputs_enc = tf.placeholder(tf.float32, [None, seq_len], name='inputs_enc')


############################ Encoder ############################
'''input --> 28, 28 (784)
c1 = conv1d(input, , name='c1')       kernel size: (5,5), n_filters:25 ???
conv1.shape (?, 10240, 8)
pool1.shape (?, 5120, 8)
conv2.shape (?, 5120, 4)
pool2.shape (?, 2560, 4)
conv3.shape (?, 2560, 4)
pool3.shape (?, 1280, 4)
dense1.shape (?, 2000)
z_1 (?, 50)
input_dec (?, 2000, 1)
dedense1.shape (?, 1280, 4)
deconv1.shape (?, 1280, 4)
deconv2.shape (?, 2560, 4)
deconv3.shape (?, 5120, 8)
depool.shape (?, 10240, 8)
reconstruction (?, 10240, 1)
'''
def encoder(inputs_enc):
    """def encoder_net(x, latent_dim, h_dim):
    Construct an inference network parametrizing a Gaussian.
    Args:
    x: A batch of real data (MNIST digits).
    latent_dim: The latent dimensionality.
    hidden_size: The size of the neural net hidden layers.
    Returns:
    mu: Mean parameters for the variational family Normal
    sigma: Standard deviation parameters for the variational family Normal
    often in convolution padding = 'same', in max_pooling padding = 'valid'
    """

    with tf.variable_scope('enc') as scope:
        inputs_enc = tf.reshape(inputs_enc, [-1,  seq_len, 1])
        print "intpus_enc", inputs_enc.shape
            
        # Convolutional Layer #1
        with tf.name_scope("conv1"):
            conv1 = tf.layers.conv1d(
                                                        inputs = inputs_enc,
                                                        filters = 8, #16,          #
                                                        kernel_size = 5,            #[3, 3],        #
                                                        padding = 'same',
                                                        activation=tf.nn.relu)
            # Pooling Layer #1
            pool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2)
            print "conv1.shape", conv1.shape   # conv1.shape (?, 5120, 32)
            print "pool1.shape", pool1.shape       # (?, 2560, 32)
            #pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], padding='same' , strides=2)
        with tf.name_scope("conv2"):
            ## Convolutional Layer #2
            conv2 = tf.layers.conv1d(
                                                        inputs = pool1,
                                                        filters = 4,
                                                        kernel_size = 5,
                                                        padding = 'same',
                                                        activation=tf.nn.relu)
            ## Pooling Layer #1
            pool2 = tf.layers.max_pooling1d(conv2, pool_size=2, padding='same', strides=2)
            print "conv2.shape", conv2.shape
            print "pool2.shape", pool2.shape
            #### Conv 3
            #conv3 = tf.layers.conv1d(
                                                        #inputs = pool2,
                                                        #filters = 4,
                                                        #kernel_size = 5,
                                                        #padding = 'same',
                                                        #activation=tf.nn.tanh)
            ### Pooling Layer #1
            #pool3 = tf.layers.max_pooling1d(conv3, pool_size=2, padding='same', strides=2)
            #print "conv3.shape", conv3.shape
            #print "pool3.shape", pool3.shape
            pool2_flat = tf.reshape(pool2, [-1,  pool2.shape[1]*pool2.shape[2]])  # (?, 81920)
            dense1 = tf.layers.dense(inputs=pool2_flat, units=hid_dim1, activation=tf.nn.tanh)
            dropout1 = tf.layers.dropout(inputs=dense1, rate=0.75, name='dec_dropout1')
            print "dense1.shape", dense1.shape

        with tf.name_scope("dense"):
            ### Get mu
            mu_1 = tf.contrib.layers.fully_connected(dropout1, latent_dim, activation_fn=None)
            # layer 2   Output mean and std of the latent variable distribution
            sigma_1 = tf.contrib.layers.fully_connected(dropout1, latent_dim, activation_fn=None)
            # Reparameterize import Randomness
            noise = tf.random_normal([1, latent_dim])
            # z_1 is the fisrt leverl output(latent variable) of our Encoder
            z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))
            print "z_1", z_1.shape
        return mu_1, sigma_1, z_1     #dense1                            #

def decoder(inputs_dec):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    inputs_dec: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
"""
    with tf.variable_scope('dec') as scope:
        print "input_dec.shape", inputs_dec.shape
        with tf.name_scope('dec_fc_dropout'):
            dedense1 = tf.layers.dense(inputs=inputs_dec, units=hid_dim1, activation=tf.nn.relu)
            print "dedense1.shape", dedense1.shape
            dropout1 = tf.layers.dropout(inputs=dedense1, rate=0.75, name='dec_dropout1')
            dedense1 = tf.layers.dense(inputs=dropout1, units=4 *  seq_len/4, activation=tf.nn.relu)
            dedense1 = tf.reshape(dedense1, [-1, seq_len/4, 4])
            ## Convolutional Layer #2
            deconv1 = tf.layers.conv1d(
                                                        inputs = dedense1,
                                                        filters = 4,
                                                        kernel_size = 5,
                                                        padding = 'same',
                                                        activation=tf.nn.relu)
            ## Pooling Layer #1
            depool1 = tf.keras.layers.UpSampling1D(2)(deconv1)
            print "deconv1.shape", deconv1.shape
            print "depool1.shape", depool1.shape
            ### Convolutional Layer #2
            #deconv2 = tf.layers.conv1d(
                                                        #inputs = depool1,
                                                        #filters = 4,
                                                        #kernel_size = 5,
                                                        #padding = 'same',
                                                        #activation=tf.nn.tanh)
            ### Pooling Layer #1
            ##depool2 = tf.keras.layers.UpSampling1D(2)(deconv2)
            #print "deconv2.shape", deconv2.shape
            #print "depool2.shape", depool2.shape
            
        # Convolutional Layer #3
            deconv3 = tf.layers.conv1d(
                                                            inputs = depool1,
                                                            filters = 8,
                                                            kernel_size = 5,
                                                            padding = 'same',
                                                            activation=tf.nn.relu)
            ## Pooling Layer #1
            depool3 = tf.keras.layers.UpSampling1D(2)(deconv3)
            print "deconv3.shape", deconv3.shape
            print "depool3.shape", depool3.shape
            ### fc
            reconstruction = tf.layers.conv1d(
                                                            inputs = depool3,
                                                            filters = 1,
                                                            kernel_size = 5,
                                                            padding = 'same',
                                                            activation=tf.nn.sigmoid) #### OMG!!! Wrong activation
            reconstruction = tf.reshape(reconstruction, [-1, seq_len])
            print "reconstruction.shape", reconstruction.shape      #reconstruction (?, 5120, 1)

        return reconstruction

def train(input_enc):
    ##### Get data
    files_train = func.find_files(data_dir, pattern="ds_8*", withlabel=True )### traverse all the files in the dir, and divide into batches, from
    files_test = func.find_files(data_dir_test, pattern="ds_8*", withlabel=True)### traverse all the files in the dir, and divide into batches, from
    file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
    file_tensor_test = tf.convert_to_tensor(files_test, dtype=tf.string)## convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
    dataset_test = tf.data.Dataset.from_tensor_slices(file_tensor_test).repeat().batch(batch_size).shuffle(buffer_size=10000)
    ## create the iterator
    iter = dataset.make_initializable_iterator()
    iter_test = dataset.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename
    ele_test = iter_test.get_next()   #you get the filename

    ### Graph
    mu_1, sigma_1, z = encoder(inputs_enc)

    reconstruction = decoder(z)

    # Loss function = reconstruction error + regularization(similar image's latent representation close)
    with tf.name_scope('loss'):
        Log_loss = tf.reduce_sum(inputs_enc  * tf.log(reconstruction + 1e-7) + (1 - inputs_enc ) * tf.log(1 - reconstruction + 1e-7), reduction_indices=1)
        KL_loss = -0.5 * tf.reduce_sum(1 + 2*sigma_1 - tf.pow(mu_1, 2) - tf.exp(2 * sigma_1), reduction_indices=1)
        VAE_loss = tf.reduce_mean(Log_loss - KL_loss)

    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_loss1', tf.reduce_mean(KL_loss))
    tf.summary.scalar('Log_loss1', tf.reduce_mean(Log_loss))
    test_loss = tf.Variable(0.0)
    test_loss_sum = tf.summary.scalar('test_loss', test_loss)

    optimizer = tf.train.AdadeltaOptimizer().minimize(-VAE_loss)

    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    # Training  init all variables and start the session!
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    sess.run(iter.initializer)
    sess.run(iter_test.initializer)
    ## Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #store value for these 3 terms so we can plot them later
    vae_loss_array = []
    test_vae_array = []
    log_loss_array = []
    KL_loss_array = []
    #iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]

    ### get the real data
    for batch in range(num_iterations):
        save_name = results_dir + '/' + "_step{}_".format( batch)
        #batch_data = np.round(mnist.train.next_batch(batch_size)[0])
        files_train =  sess.run(ele)   # name, '1'/'0'
        files_test =  sess.run(ele_test)   # name, '1'/'0'
        batch_data = np.empty([0, seq_len ])
        for ind in range(len(files_train)):
            data = np.average(func.read_data(files_train[ind][0]), axis=0)
            batch_data = np.vstack((batch_data, data))

        #run our optimizer on our data
        _, summary = sess.run([optimizer, summaries], feed_dict={inputs_enc: batch_data})
        writer.add_summary(summary, batch)
        ### test
        if (batch % test_every == 0):
            #test_data = mnist.test.images[0:200]
            test_data = np.empty([0, seq_len])
            for ind in range(len(files_test)):
                data = np.average(func.read_data(files_test[ind][0]), axis=0)
                test_data = np.vstack((test_data, data))

            test_temp = VAE_loss.eval({input_enc : test_data})
            test_vae_array = np.append(test_vae_array, test_temp)

            summary = sess.run(test_loss_sum, {test_loss: test_temp})    ## add test score to summary
            writer.add_summary(summary, batch % test_every)
            #every 1K iterations record these values
            temp_vae = VAE_loss.eval(feed_dict={inputs_enc: batch_data})
            temp_log = np.mean(Log_loss.eval(feed_dict={inputs_enc: batch_data}))
            temp_KL = np.mean(KL_loss.eval(feed_dict={inputs_enc: batch_data}))
            vae_loss_array.append(temp_vae )
            KL_loss_array.append(temp_KL)
            log_loss_array.append( temp_log)
        if batch % print_result == 0:
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term {}".format(batch, temp_vae, temp_log, temp_KL )

        if (batch % save_every == 0):
            saver.save(sess, logdir + '/' + str(batch))

        if (batch % plot_every == 0):
            plt.figure()
            plt.plot(np.arange(len(vae_loss_array)), vae_loss_array, color = 'orchid', label='vae_los')
            plt.plot(np.arange(len(vae_loss_array)),  KL_loss_array, color = 'c', label='KL_loss')
            plt.plot(np.arange(len(vae_loss_array)),  log_loss_array, color = 'b', label='log_likelihood')
            plt.xlabel("training ")
            plt.legend(loc="best")
            plt.savefig(save_name+"losses_iter{}.png".format(batch), format="png")
            plt.title('Loss during training')
            plt.close()
            plt.figure()
            plt.plot(np.arange(len(test_vae_array)),  test_vae_array, color = 'darkcyan')
            plt.title('Loss in test')
            plt.savefig(save_name+"test_loss_iter{}.png".format(batch), format="png")
            plt.close()
            #func.plot_learning_curve(test_vae_array, test_vae_array, num_trial=1, save_name=resultdir + "/learning_curve.png")
            #func.plot_smooth_shadow_curve(np.array(vae_loss_array), save_name=results_dir + "/loss_in_training.png")
    func.save_data((vae_loss_array, KL_loss_array, log_loss_array), header='vae_loss,KL_loss,log_loss', save_dir=save_name+'3losses.csv')

        
if __name__ == "__main__":
    train(inputs_enc)

'''
Iteration: 0, Loss: -1559.66674805, log_loss: -785.173950195, KL_term 624.776916504
Iteration: 600, Loss: -1406.72229004, log_loss: -969.086486816, KL_term 423.029876709
Iteration: 1000, Loss: -1250.46850586, log_loss: -941.946777344, KL_term 305.732299805
Iteration: 2000, Loss: -1062.35107422, log_loss: -886.799804688, KL_term 175.553817749
Iteration: 3000, Loss: -978.892456055, log_loss: -797.363830566, KL_term 189.481140137
Iteration: 4000, Loss: -862.5078125, log_loss: -789.658935547, KL_term 102.668769836
Iteration: 5000, Loss: -1050.76171875, log_loss: -996.399780273, KL_term 103.408706665'''
