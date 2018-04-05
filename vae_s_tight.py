import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time #lets clock training time..
import os
from scipy.misc import imsave
#import data
#from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets    #read data
import ipdb
import datetime

SAVE_EVERY = 20000
plot_every = 5000
version = "EEG_ds16"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
logdir = "model/" + version + '/' + datetime
results_dir = "results/" + version + '/' + datetime
data_dir = "data/sub_train/sub_16"
data_dir_test = "data/sub_test"
n_pixels =  640              ####328 * 28
batch_size = 100
# HyperParameters
latent_dim = 50
h_dim1 = 500  # size of network
h_dim2 = 200  # size of network
num_iterations = 1000001   # 50
recording_interval = 1000    # 1000   #
# load data
#mnist = read_data_sets(data_dir, one_hot=True)
def get_test_data():
    test_data = np.empty([0, in_dim])
    test_labels = np.empty([0])
    for filen in files_test:
        data = np.average(func.read_data(filen[0]), axis=0)
        test_data = np.vstack((test_data, data))                
        test_labels = np.append(test_labels, filen[1])
    test_labels = np.eye((n_classes))[test_labels.astype(int)]
    return test_data, test_labels

############################ Encoder ############################
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
    """
    
    with tf.variable_scope('enc') as scope:
        # layer 1
        enc_hidden1 = tf.contrib.layers.fully_connected(
                                                                            inputs_enc,  # input data
                                                                            h_dim1,    # output dimension
                                                                            activation_fn=tf.nn.tanh)   # activation function
        enc_hidden2 = tf.contrib.layers.fully_connected(
                                                                            enc_hidden1,
                                                                            h_dim2,
                                                                            activation_fn=tf.nn.tanh)
        mu_1= tf.contrib.layers.fully_connected(
                                                                            enc_hidden2,
                                                                            latent_dim,
                                                                            activation_fn=None)
        sigma_1= tf.contrib.layers.fully_connected(
                                                                            enc_hidden2,
                                                                            latent_dim,
                                                                            activation_fn=None)
        return mu_1, sigma_1

def decoder(inputs_dec):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    z: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
    """

    with tf.variable_scope('dec') as scope:
        # layer 1
        dec_hidden2 = tf.contrib.layers.fully_connected(
                                                                            inputs_dec,
                                                                            h_dim2,
                                                                            activation_fn=tf.nn.tanh)
        dec_hidden1 = tf.contrib.layers.fully_connected(
                                                                            dec_hidden2,
                                                                            h_dim1,
                                                                            activation_fn=tf.nn.tanh)

        reconstruction = tf.contrib.layers.fully_connected(
                                                                            dec_hidden2,
                                                                            n_pixels,
                                                                            activation_fn=tf.nn.sigmoid)
        return reconstruction


def train():
    #### Get data
    files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    files_test = func.find_files(data_dir_test, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
    ## create the iterator
    iter = dataset.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename
    
    with tf.name_scope("input"):
        #real and fake image placholders
        inputs_enc = tf.placeholder(tf.float32, shape = [None, n_pixels], name='inputs_enc')
        #real_image = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
        inputs_dec = tf.placeholder(tf.float32, shape=[None, latent_dim], name='rand_input')

    mu, logstd = encoder(inputs_enc)
    ##Repsarameterize import Randomness
    noise = tf.random_normal([1, latent_dim])
     ##z_1 is the fisrt leverl output(latent variable) of our Encoder
    z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
    reconstruction = decoder(z)
    
    with tf.name_scope('loss'):
        log_likelihood = tf.reduce_sum(inputs_enc * tf.log(reconstruction + 1e-9) + (1 - inputs_enc) * tf.log(1 - reconstruction + 1e-9))

        KL_divergence = -0.5 * tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) - tf.exp(2 * logstd), reduction_indices=1)

        VAE_loss = tf.reduce_mean(log_likelihood + KL_divergence)
        

    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_divergence1', KL_divergence)
    tf.summary.scalar('log_likelihood1', log_likelihood)

    optimizer = tf.train.AdadeltaOptimizer().minimize(-VAE_loss)

    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    # Training
    #init all variables and start the session!
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    ## Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    num_iterations = 1000001   # 50
    recording_interval = 1000    # 1000   #
    #store value for these 3 terms so we can plot them later
    variational_lower_bound_array = []
    log_likelihood_array = []
    KL_term_array = []
    iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]

    for i in range(num_iterations):
        save_name = results_dir + '/' + "_step{}_".format(i)
        # np.round to make MNIST binary
        #get first batch (200 digits)
        x_batch = np.round(mnist.train.next_batch(batch_size)[0])
        #run our optimizer on our data
        sess.run(optimizer, feed_dict={inputs_enc: x_batch})
        if (i % recording_interval == 0):
            #every 1K iterations record these values
            vlb_eval = VAE_loss.eval(feed_dict={inputs_enc: x_batch})
            
            variational_lower_bound_array.append(vlb_eval)
            temp_log = np.mean(log_likelihood.eval(feed_dict={inputs_enc: x_batch}))
            log_likelihood_array.append(temp_log)
            temp_KL = np.mean(KL_divergence.eval(feed_dict={inputs_enc: x_batch}))
            KL_term_array.append(temp_KL)
            print "Iteration: {}, Loss: {}, log_likelihood: {}, KL_term{}".format(i, vlb_eval, temp_log, temp_KL )


if __name__ == "__main__":
    train()
    #if (i % 10 == 0):
        #if not os.path.exists(logdir):
            #os.makedirs(logdir)
        #saver.save(sess, logdir + '/' + str(i))
        

    #if (i % plot_every == 0):
        ##plot_prior(model_No)
        #plot_test(i, save_name=save_name)
        
        #plt.figure()
        ##for the number of iterations we had 
        ##plot these 3 terms
        #plt.plot(iteration_array, variational_lower_bound_array)
        #plt.plot(iteration_array, KL_term_array)
        #plt.plot(iteration_array, log_likelihood_array)
        #plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
        #plt.title('Loss per iteration')
        #plt.savefig(save_name+"_iter{}_loss.png".format(i), format="png")
        


'''
Iteration: 0, Loss: -122970.8125, log_likelihood: -120843.765625, KL_term213.671951294'''





