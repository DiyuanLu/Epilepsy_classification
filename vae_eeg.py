
#### This is a script treat the eeg data(2250 clips) as sequence data and apply VAE to encode feateures
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
#matplotlib.use('Agg')
import os
from scipy.misc import imsave
#import data
from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
import ipdb
import datetime
import functions as func
mnist = input_data.read_data_sets("data/MNIST_data/",  one_hot=True)


SAVE_EVERY = 20000
plot_every = 5000
version = "EEG_ds16"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
logdir = "model/" + version + '/' + datetime
results_dir = "results/" + version + '/' + datetime
data_dir = "data/sub_train/sub_16"
data_dir_test = "data/sub_test"

num_iterations = 1000001   # 50
recording_interval = 1000    # 1000   #
### Hyperparams
in_dim = 28 * 28#640
hid_dim1 = 1000   # Encoder: input -- hidden1 -- latent1 -- hidden2 -- latent2 
hid_dim2 = 500
latent_dim = 50
batch_size = 100

def get_test_data():
    test_data = np.empty([0, in_dim])
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

############################ Encoder ############################
def encoder(real_data):
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
    h_dim_1, latent_dim_1, h_dim_2, latent_dim_2 = 1000, 200, 300, 50 #
    
    with tf.variable_scope('enc') as scope:
        # layer 1
        W_enc_1 = weight_variables([n_pixels, h_dim_1], "W_enc_1")
        b_enc_1 = bias_variable([h_dim_1], "b_enc_1")
        # tanh - activation function        avoid vanishing gradient in generative models
        h_enc_1 = tf.nn.tanh(FC_Layer(real_data, W_enc_1, b_enc_1))

        # layer 2   Output mean and std of the latent variable distribution
        W_mu_1 = weight_variables([h_dim_1, latent_dim_1], "W_mu_1")
        b_mu_1 = bias_variable([latent_dim_1], "b_mu_1")
        mu_1 = FC_Layer(h_enc_1, W_mu_1, b_mu_1)

        W_logstd_1 = weight_variables([h_dim_1, latent_dim_1], "W_logstd_1")
        b_logstd_1 = bias_variable([latent_dim_1], "b_logstd_1")
        logstd_1 = FC_Layer(h_enc_1, W_logstd_1, b_logstd_1)

        # Reparameterize import Randomness
        noise = tf.random_normal([1, latent_dim_1])
        # z_1 is the fisrt leverl output(latent variable) of our Encoder
        z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*logstd_1))

        # second level---------------------------- layer 1
        W_enc_2 = weight_variables([latent_dim_1, h_dim_2], "W_enc_2")
        b_enc_2 = bias_variable([h_dim_2], "b_enc_2")
        # tanh - activation function        avoid vanishing gradient in generative models
        h_enc_2 = tf.nn.tanh(FC_Layer(z_1, W_enc_2, b_enc_2))

        # layer 2   Output mean and std of the latent variable distribution
        W_mu_2 = weight_variables([h_dim_2, latent_dim_2], "W_mu_2")
        b_mu_2 = bias_variable([latent_dim_2], "b_mu_2")
        mu_2 = FC_Layer(h_enc_2, W_mu_2, b_mu_2)

        W_logstd_2 = weight_variables([h_dim_2, latent_dim_2], "W_logstd_2")
        b_logstd_2 = bias_variable([latent_dim_2], "b_logstd_2")
        logstd_2 = FC_Layer(h_enc_2, W_logstd_2, b_logstd_2)

        # Reparameterize import Randomness
        noise_2 = tf.random_normal([1, latent_dim_2])
        # z_1 is the ultimate output(latent variable) of our Encoder
        z_2 = mu_2 + tf.multiply(noise_2, tf.exp(0.5*logstd_2))

        return mu_1, logstd_1, mu_2, logstd_2, z_1

############################ Dencoder ############################
def decoder(random_input, z_1):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    z: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
    """
    h_dim_1, latent_dim_1, h_dim_2, latent_dim_2 = 1000, 200, 300, 50 # channel num

    with tf.variable_scope('dec') as scope:
        # layer 1
        W_dec_2 = weight_variables([latent_dim_2, h_dim_2], "W_dec_2")
        b_dec_2 = bias_variable([h_dim_2], "b_dec_2")
        # tanh - decode the latent representation
        h_dec_2 = tf.nn.tanh(FC_Layer(random_input, W_dec_2, b_dec_2))

        # layer2 - reconstruction the first leverl latent variables
        W_rec_2 = weight_variables([h_dim_2, latent_dim_1], "W_dec_2")
        b_rec_2 = bias_variable([latent_dim_1], "b_rec_2")
        recon_z1 = tf.nn.sigmoid(FC_Layer(h_dec_2, W_rec_2, b_rec_2)) # ?????

        # layer 1
        W_dec_1 = weight_variables([latent_dim_1, h_dim_1], "W_dec")
        b_dec_1 = bias_variable([h_dim_1], "b_dec")
        # tanh - decode the latent representation

        #ipdb.set_trace()
        residual_z1 = tf.identity(z_1) + recon_z1
        h_dec_1 = tf.nn.tanh(FC_Layer(residual_z1 , W_dec_1, b_dec_1))

        # layer2 - reconstruction the image and output 0 or 1
        W_rec_1 = weight_variables([h_dim_1, n_pixels], "W_rec_1")
        b_rec_1 = bias_variable([n_pixels], "b_rec_1")
        # 784 bernoulli parameter Output
        reconstruction = tf.nn.sigmoid(FC_Layer(h_dec_1, W_rec_1, b_rec_1))

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
    #z: Samples of latent variables with size latent_dim_2
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
                                                                            #in_dim,
                                                                            #activation_fn=tf.nn.sigmoid)
        #return reconstruction

def train():
    #### Get data
    files_train = func.find_files(data_dir, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    files_test = func.find_files(data_dir_test, withlabel=True )### traverse all the files in the dir, and divide into batches, from
    file_tensor_train = tf.convert_to_tensor(files_train, dtype=tf.string)## convert to tensor
    dataset = tf.data.Dataset.from_tensor_slices(file_tensor_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
    ## create the iterator
    iter = dataset.make_initializable_iterator()
    ele = iter.get_next()   #you get the filename

    ##### input
    with tf.name_scope("input"):
        ### the real data from database
        inputs_enc = tf.placeholder(tf.float32, [None, in_dim], name='inputs_enc')
        ### random noise to reconstruct the real data
        inputs_dec = tf.placeholder(tf.float32, [None, latent_dim1], name='inputs_dec')
        
    ### Graph    
    mu_1, sigma_1 = encoder(inputs_enc)
    ##Repsarameterize import Randomness
    noise = tf.random_normal([1, latent_dim])
     ##z_1 is the fisrt leverl output(latent variable) of our Encoder
    inputs_dec = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))
    reconstruction = decoder(inputs_dec)
    
    # Loss function = reconstruction error + regularization(similar image's latent representation close)
    with tf.name_scope('loss'):
        log_loss = tf.reduce_sum(inputs_enc  * tf.log(reconstruction + 1e-9) + (1 - inputs_enc ) * tf.log(1 - reconstruction + 1e-9))

        KL_loss = -0.5 * tf.reduce_sum(1 + 2*sigma_1 - tf.pow(mu_1, 2) - tf.exp(2 * sigma_1), reduction_indices=1)

        VAE_loss = tf.reduce_mean(log_loss + KL_loss)
        
    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_loss1', KL_loss)
    tf.summary.scalar('log_loss1', log_loss)

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
    sess.run(iter.initializer)
    ## Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #store value for these 3 terms so we can plot them later
    variational_lower_bound_array = []
    log_loss_array = []
    KL_loss_array = []
    iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]

    ### get the real data
    
    
    for i in range(num_iterations):
        train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_dim1]).astype(np.float32)
        filename =  sess.run(ele)   # name, '1'/'0'
        batch_data = np.empty([0, in_dim ])
        for ind in range(len(filename)):
            data = np.average(func.read_data(filename[ind][0]), axis=0)
            batch_data = np.vstack((batch_data, data))
            ipdb.set_trace()          
        #batch_data = np.round(mnist.train.next_batch(batch_size)[0])
        save_name = results_dir + '/' + "_step{}_".format(i)
        #run our optimizer on our data
        _, reconstruction_batch, log_loss_batch, KL_loss_batch, VAE_loss_batch = sess.run([optimizer, reconstruction, log_loss, KL_loss, VAE_loss], feed_dict={inputs_enc: batch_data, inputs_dec : train_noise})
        
        if (i % 100 == 0):
            #every 1K iterations record these values
            vlb_eval = VAE_loss.eval(feed_dict={inputs_enc: batch_data, inputs_dec : train_noise})
            
            variational_lower_bound_array.append(VAE_loss_batch)
            temp_log = np.mean(log_loss.eval(feed_dict={inputs_enc: batch_data, inputs_dec : train_noise}))
            log_loss_array.append( log_loss_batch)
            temp_KL = np.mean(KL_loss.eval(feed_dict={inputs_enc: batch_data, inputs_dec : train_noise}))
            KL_loss_array.append(KL_loss_batch)
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term{}".format(i, vlb_eval, temp_log, temp_KL )

        #if (i % 10000 == 0):
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
            #plt.plot(iteration_array, KL_loss_array)
            #plt.plot(iteration_array, log_loss_array)
            #plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
            #plt.title('Loss per iteration')
            #plt.savefig(save_name+"_iter{}_loss.png".format(i), format="png")

if __name__ == "__main__":
    train()
