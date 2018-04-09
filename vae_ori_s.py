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

save_every = 2000
plot_every = 5000
num_iterations = 100001   # 50
recording_interval = 100    # 1000   #
n_pixels = 28 * 28
# HyperParameters
latent_dim = 20
h_dim = 500  # size of network
version = "MNIST"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
data_dir = "training_data/MNIST_data"
logdir = "results/" + version + '/' + datetime + "/model"
results_dir = "results/" + version + '/' + datetime
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir ):
    os.makedirs(results_dir )
    
# load data
mnist = read_data_sets(data_dir, one_hot=True)

# input the image
X = tf.placeholder(tf.float32, shape=([None, n_pixels]))
#Z = tf.placeholder(tf.float32, shape=([None, latent_dim]))

def weight_variables(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def FC_Layer(X, W, b):
    return tf.matmul(X, W) + b

def plot_test(model_No, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))

    num_pairs = 10
    image_indices = np.random.randint(0, 200, num_pairs)
    #Lets plot 10 digits
    
    for pair in range(num_pairs):
        #reshaping to show original test image
        x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
        x_image = np.reshape(x, (28,28))
        
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(x_image, aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, 27])
        plt.ylim([27, 0])
        
        #reconstructed image, feed the test image to the decoder
        x_reconstruction = reconstruction.eval(feed_dict={X: x})
        #reshape it to 28x28 pixels
        x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
        #plot it!
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, 27])
        plt.ylim([27, 0])
        plt.tight_layout()
        if pair == 0 or pair == 1:
            plt.title("Reconstruct")
    #ipdb.set_trace()
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")

def plot_prior(model_No):
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    nx = ny = 5     
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((28 * ny, 28 * nx))
    noise = tf.random_normal([1, 20])
    z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
    ipdb.set_trace()
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        z[0:2] = np.array([[xi, yi]])
        x_reconstruction = reconstruction.eval(feed_dict={z: z})
        ## layer 1
        #W_dec = weight_variables([latent_dim, h_dim], "W_dec")
        #b_dec = bias_variable([h_dim], "b_dec")
        ## tanh - decode the latent representation
        #h_dec = tf.nn.tanh(FC_Layer(z, W_dec, b_dec))

        ## layer2 - reconstruction the image and output 0 or 1
        #W_rec = weight_variables([h_dim, n_pixels], "W_dec")
        #b_rec = bias_variable([n_pixels], "b_rec")
        ## 784 bernoulli parameter Output
        #reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
    
        canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
               28:(j + 1) * 28] = reconstruction[0].reshape(28, 28)
    imsave(os.path.join(logdir,
                        'prior_predictive_map_frame_%d.png' % model_No), canvas)
############################ Encoder ############################


def encoder(X):
    # layer 1
    with tf.name_scope('Encoder'):
        W_enc = weight_variables([n_pixels, h_dim], "W_enc")
        b_enc = bias_variable([h_dim], "b_enc")
        # tanh - activation function        avoid vanishing gradient in generative models
        h_enc = tf.nn.tanh(FC_Layer(X, W_enc, b_enc))

        # layer 2   Output mean and std of the latent variable distribution
        W_mu = weight_variables([h_dim, latent_dim], "W_mu")
        b_mu = bias_variable([latent_dim], "b_mu")
        mu = FC_Layer(h_enc, W_mu, b_mu)

        W_logstd = weight_variables([h_dim, latent_dim], "W_logstd")
        b_logstd = bias_variable([latent_dim], "b_logstd")
        logstd = FC_Layer(h_enc, W_logstd, b_logstd)


        # Reparameterize import Randomness
        noise = tf.random_normal([1, latent_dim])
        # z is the ultimate output(latent variable) of our Encoder
        z = mu + tf.multiply(noise, tf.exp(0.5*logstd))
        return mu, logstd, z

############################ Dencoder ############################
def decoder(z):
    '''Z: random_input 1d(latent_dim,)'''
    # layer 1
    with tf.name_scope('Decoder'):
        W_dec = weight_variables([latent_dim, h_dim], "W_dec")
        b_dec = bias_variable([h_dim], "b_dec")
        # tanh - decode the latent representation
        h_dec = tf.nn.tanh(FC_Layer(z, W_dec, b_dec))

        # layer2 - reconstruction the image and output 0 or 1
        W_rec = weight_variables([h_dim, n_pixels], "W_dec")
        b_rec = bias_variable([n_pixels], "b_rec")
        # 784 bernoulli parameter Output
        reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
        return reconstruction

# Loss function = reconstruction error + regularization(similar image's latent representation close)
def train(X):
    mu, logstd, z = encoder(X)
    reconstruction = decoder(z)
    with tf.name_scope('loss'):
        Log_loss = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9))
        KL_loss = -0.5 * tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) - tf.exp(2 * logstd), reduction_indices=1)
        VAE_loss = tf.reduce_mean(Log_loss + KL_loss)
        
    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_loss', tf.reduce_mean(KL_loss))
    tf.summary.scalar('Log_loss',  tf.reduce_mean(Log_loss))

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

    #store value for these 3 terms so we can plot them later
    variational_lower_bound_array = []
    log_loss_array = []
    KL_term_array = []
    iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]

    for i in range(num_iterations):
        save_name = results_dir + '/' + "_step{}_".format(i)
        # np.round to make MNIST binary
        #get first batch (200 digits)
        x_batch = np.round(mnist.train.next_batch(200)[0])
        #run our optimizer on our data
        _, summary = sess.run([optimizer, summaries], feed_dict={X: x_batch})
        writer.add_summary(summary, batch)
        if (i % recording_interval == 0):
            #every 1K iterations record these value
            temp_vae = np.mean(VAE_loss.eval(feed_dict={X: x_batch}))
            variational_lower_bound_array.append(temp_vae)
            temp_log = np.mean(Log_loss.eval(feed_dict={X: x_batch}))
            log_loss_array.append(temp_log)
            temp_KL = np.mean(KL_loss.eval(feed_dict={X: x_batch}))
            KL_term_array.append(temp_KL)
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term{}".format(i, temp_vae, temp_log, temp_KL )

        if (i % save_every == 0):
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            saver.save(sess, logdir + '/' + str(i))
            

        #if (i % plot_every == 0):
            ##plot_prior(model_No)
            #plot_test(i, save_name=save_name)
            
            #plt.figure()
            ##for the number of iterations we had 
            ##plot these 3 terms
            #plt.plot(iteration_array, variational_lower_bound_array)
            #plt.plot(iteration_array, KL_term_array)
            #plt.plot(iteration_array, log_loss_array)
            #plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
            #plt.title('Loss per iteration')
            #plt.savefig(save_name+"_iter{}_loss.png".format(i), format="png")

if __name__ == "__main__":
    train(X)
