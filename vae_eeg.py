
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
from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
mnist = input_data.read_data_sets("data/MNIST_data/",  one_hot=True)


SAVE_EVERY = 20000
plot_every = 5000
version = "vae_eeg_MNIST_bn"     ###"VAE_ds16"  #
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir = "results/" + version + '/' + datetime
logdir = results_dir  + "/model" 

data_dir = "data/sub_train/sub_16"
data_dir_test = "data/sub_test/sub_16"

num_iterations = 1000001   # 50
recording_interval = 1000    # 1000   #
### Hyperparams
seq_len = 28 * 28#   640     #
hid_dim1 = 1000   # Encoder: input -- hidden1 -- latent1 -- hidden2 -- latent2 
hid_dim2 = 500
latent_dim = 50
batch_size = 200

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

#### input
with tf.name_scope("input"):
    ## the real data from database
    inputs_enc = tf.placeholder(tf.float32, [None, seq_len], name='inputs_enc')
    ## random noise to reconstruct the real data
    #inputs_dec = tf.placeholder(tf.float32, [None, latent_dim1], name='inputs_dec')

############################ Encoder ############################
def encoder(inputs_enc):
    
    with tf.name_scope('Encoder'):
        # layer 1
        W_enc = weight_variables([seq_len, hid_dim2], "W_enc")
        b_enc = bias_variable([hid_dim2], "b_enc")
        # tanh - activation function        avoid vanishing gradient in generative models
        fc = FC_Layer(inputs_enc, W_enc, b_enc)
        bn_fc = tf.layers.batch_normalization(fc, center=True, scale=True)
        h_enc = tf.nn.tanh(bn_fc)

        # layer 2   Output mean and std of the latent variable distribution
        W_mu = weight_variables([hid_dim2, latent_dim], "W_mu")
        b_mu = bias_variable([latent_dim], "b_mu")
        mu = FC_Layer(h_enc, W_mu, b_mu)

        W_logstd = weight_variables([hid_dim2, latent_dim], "W_logstd")
        b_logstd = bias_variable([latent_dim], "b_logstd")
        logstd = FC_Layer(h_enc, W_logstd, b_logstd)


        # Reparameterize import Randomness
        noise = tf.random_normal([1, latent_dim])
        # z is the ultimate output(latent variable) of our Encoder
        z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
        return mu, logstd, z

############################ Dencoder ############################
def decoder(inputs_dec):
    '''Z: random_input 1d(latent_dim,)'''
    # layer 1
    with tf.name_scope('Decoder'):
        W_dec = weight_variables([latent_dim, hid_dim2], "W_dec")
        b_dec = bias_variable([hid_dim2], "b_dec")

        fc_dec = FC_Layer(inputs_dec, W_dec, b_dec)
        bn_fc_dec = tf.layers.batch_normalization(fc_dec, center=True, scale=True)
        # tanh - decode the latent representation
        h_dec= tf.nn.tanh(bn_fc_dec)
    
        # layer2 - reconstruction the image and output 0 or 1
        W_rec = weight_variables([hid_dim2, seq_len], "W_dec")
        b_rec = bias_variable([seq_len], "b_rec")
        # 784 bernoulli parameter Output
        reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
        return reconstruction
  
## Variational Autoencoder
#def encoder(inputs_enc):
    #"""def encoder_net(x, latent_dim, h_dim):
    #Construct an inference network parametrizing a Gaussian.
    #Args:
    #x: A batch of real data (MNIST digits).
    #latent_dim: The latent dimensionality.
    #hidden_size: The size of the neural net hidden layers.
    #Returns:
    #mu: Mean parameters for the variational family Normal
    #sigma: Standard deviation parameters for the variational family Normal
    #"""
    
    #with tf.variable_scope('enc') as scope:
        ## layer 1
        #enc_hidden1 = tf.contrib.layers.fully_connected(
                                                                            #inputs_enc,  # input data
                                                                            #hid_dim1,    # output dimension
                                                                            #activation_fn=tf.nn.tanh)   # activation function
        #enc_hidden2 = tf.contrib.layers.fully_connected(
                                                                            #enc_hidden1,
                                                                            #hid_dim2,
                                                                            #activation_fn=tf.nn.tanh)
        #mu_1= tf.contrib.layers.fully_connected(
                                                                            #enc_hidden2,
                                                                            #latent_dim1,
                                                                            #activation_fn=None)
        #sigma_1= tf.contrib.layers.fully_connected(
                                                                            #enc_hidden2,
                                                                            #latent_dim1,
                                                                            #activation_fn=None)
        ## Reparameterize import Randomness
        #noise = tf.random_normal([1, latent_dim1])
        ## z_1 is the fisrt leverl output(latent variable) of our Encoder
        #z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))

        #return mu_1, sigma_1


#def decoder(inputs_dec):
    #"""Build a generative network parametrizing the likelihood of the data
    #Args:
    #inputs_dec: Samples of latent variables with size latent_dim_2
    #hidden_size: Size of the hidden state of the neural net
    #Returns:
    #reconstruction: logits for the Bernoulli likelihood of the data
    #"""

    #with tf.variable_scope('dec') as scope:
        ## layer 1
        #dec_hidden2 = tf.contrib.layers.fully_connected(
                                                                            #inputs_dec,
                                                                            #hid_dim2,
                                                                            #activation_fn=tf.nn.tanh)
        #dec_hidden1 = tf.contrib.layers.fully_connected(
                                                                            #dec_hidden2,
                                                                            #hid_dim1,
                                                                            #activation_fn=tf.nn.tanh)

        #reconstruction = tf.contrib.layers.fully_connected(
                                                                            #dec_hidden1,
                                                                            #seq_len,
                                                                            #activation_fn=tf.nn.sigmoid)
        #return reconstruction

def train(input_enc):
    #### Get data
    #files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    #files_test = func.find_files(data_dir_test, withlabel=True)### traverse all the files in the dir, and divide into batches, from
    #file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
    #dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
    ### create the iterator
    #iter = dataset.make_initializable_iterator()
    #ele = iter.get_next()   #you get the filename
        
    ### Graph    
    mu_1, sigma_1, z = encoder(inputs_enc)
    reconstruction = decoder(z)
    
    # Loss function = reconstruction error + regularization(similar image's latent representation close)
    with tf.name_scope('loss'):
        Log_loss = tf.reduce_sum(inputs_enc  * tf.log(reconstruction + 1e-9) + (1 - inputs_enc ) * tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
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
    # Training
    #init all variables and start the session!
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    #sess.run(iter.initializer)
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
        batch_data = np.round(mnist.train.next_batch(batch_size)[0])
        #filename =  sess.run(ele)   # name, '1'/'0'
        #batch_data = np.empty([0, seq_len ])
        #for ind in range(len(filename)):
            #data = np.average(func.read_data(filename[ind][0]), axis=0)
            #batch_data = np.vstack((batch_data, data))      
        #batch_data = np.round(mnist.train.next_batch(batch_size)[0])
        save_name = results_dir + '/' + "_step{}_".format( batch)
        #run our optimizer on our data
        _, summary = sess.run([optimizer, summaries], feed_dict={inputs_enc: batch_data})
        writer.add_summary(summary,  batch)

        ### test
        if (batch % 100 == 0):
            test_data = mnist.test.images[0:200]
            ##test_data = np.empty([0, seq_len])
            ##for ind in range(len(files_test)):
                ##data = np.average(func.read_data(files_test[ind][0]), axis=0)
                ##test_data = np.vstack((test_data, data))
            vae_temp = VAE_loss.eval({input_enc : test_data})
            test_vae_array = np.append(test_vae_array, vae_temp)
            summary = sess.run(test_loss_sum, {test_loss: vae_temp})    ## add test score to summary
            writer.add_summary(summary, batch)
            #every 1K iterations record these values
            temp_vae = VAE_loss.eval(feed_dict={inputs_enc: batch_data})
            temp_log = np.mean(Log_loss.eval(feed_dict={inputs_enc: batch_data}))
            temp_KL = np.mean(KL_loss.eval(feed_dict={inputs_enc: batch_data}))
            vae_loss_array.append(temp_vae )
            KL_loss_array.append(temp_KL)
            log_loss_array.append( temp_log)
        if batch % 200 == 0:
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term{}".format(batch, temp_vae, temp_log, temp_KL )
        
        if (batch % 100 == 0):
            saver.save(sess, logdir + '/' + str(batch))
            
        if (batch % 1000 == 0):
            ##plot_prior(model_No)
            #plot_test(batch, save_name=save_name)
            
            plt.figure()
            plt.plot(np.arange(len(vae_loss_array)), vae_loss_array)
            plt.plot(np.arange(len(KL_loss_array)), KL_loss_array)
            plt.plot( np.arange(len(log_loss_array)), log_loss_array)
            plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], loc="best")
            plt.title('Loss per batch')
            plt.savefig(save_name+"loss_iter{}_loss.png".format(batch), format="png")
            plt.close()

if __name__ == "__main__":
    train(inputs_enc)
