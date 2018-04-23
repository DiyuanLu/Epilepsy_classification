import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time #lets clock training time..
import os
import Image
#import data
#from tensorflow.examples.tutorials.mnist import input_data    # DOWNLOAD DATA
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets    #read data
import ipdb
import datetime

save_every = 5000
plot_every = 500
epochs = 50
num_iterations = epochs * 55000 + 1   # 50
record_every = 1000    # 1000   #
test_every = 1000   # check the loss on test set
n_pixels = 28 * 28
height, width = 28, 28
# HyperParameters
latent_dim = 2
h_dim = 256  # size of network
batch_size = 100
version = "vae_ori_MNIST"
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
data_dir = "training_data/MNIST_data"
logdir = "results/" + version + '/' + datetime + "/model"
results_dir = "results/" + version + '/' + datetime
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir ):
    os.makedirs(results_dir )
print results_dir
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

def plot_test(original, reconstruction, load_model = False, save_name="save"):
    # Here, we plot the reconstructed image on test set images.
    #if load_model:
        #saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    num_pairs = 10
    for pair in range(num_pairs):
        #reshaping to show original test image
        x_image = np.reshape(original[pair, :], (28,28))
        index = (1 + pair) * 2
        ax1 = plt.subplot(5,4,index - 1)  # arrange in 5*4 layout
        plt.imshow(x_image, aspect="auto")
        if pair == 0 or pair == 1:
            plt.title("Original")
        plt.xlim([0, 27])
        plt.ylim([27, 0])

        x_reconstruction_image = np.reshape(reconstruction[pair, :], (28,28))
        ax2 = plt.subplot(5,4,index, sharex = ax1, sharey=ax1)
        plt.imshow(x_reconstruction_image, aspect="auto")
        plt.setp(ax2.get_yticklabels(), visible=False)
        plt.xlim([0, 27])
        plt.ylim([27, 0])
        plt.tight_layout()
        if pair == 0 or pair == 1:
            plt.title("Reconstruct")
            
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                wspace=0.30, hspace=0.22)
    plt.savefig(save_name + "samples.png", format="png")
    plt.close()
   
def plot_prior(model_No):
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    nx = ny = 5     
    x_values = np.linspace(0.05, 0.95, nx)
    y_values = np.linspace(0.05, 0.95, ny)

    f, axes = plt.subplots(nx, ny, sharex=True, sharey=True)
    for ind, ax in enumerate(axes):
        ax.imshow()
    ax1.plot(x, y)
    ax1.set_title('Sharing both axes')
    ax2.scatter(x, y)
    ax3.scatter(x, 2 * y ** 2 - 1, color='r')
    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


    canvas = np.empty((28 * ny, 28 * nx))
    noise = tf.random_normal([1, 20])
    z = mu + tf.multiply(noise, tf.exp(0.5*sigma))
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        z[0:2] = np.array([[xi, yi]])
        x_reconstruction = reconstruction.eval(feed_dict={z: z})

        canvas[(nx - ii - 1) * 28:(nx - ii) * 28, j *
               28:(j + 1) * 28] = reconstruction[0].reshape(28, 28)
    imsave(os.path.join(logdir,
                        'prior_predictive_map_frame_%d.png' % model_No), canvas)
############################ Encoder ############################
def encoder(X):
    '''fully  connected encoder with residual connections
    X: [batch, height*width]'''
    # layer 1
    with tf.name_scope('Encoder'):
        net = tf.layers.dense(inputs=X, units=h_dim, activation=tf.nn.relu)
        #### high-way net
        #H = tf.layers.dense(net, units=300, activation=tf.nn.relu, name="enH1")
        #T = tf.layers.dense(net, units=300, activation=tf.nn.sigmoid, name="enT1")
        #C = 1. - T
        #net = H * T + net * C
        #####
        #net = tf.contrib.layers.fully_connected(net, 128, activation_fn=tf.nn.tanh)
        ##### high-way net
        #H = tf.layers.dense(net, units=128, activation=tf.nn.relu, name="enH2")
        #T = tf.layers.dense(net, units=128, activation=tf.nn.sigmoid, name="enT2")
        #C = 1. - T
        #net = H * T + net * C
        ####
        mu = tf.layers.dense(inputs=net, units=latent_dim, activation=None)
        sigma = tf.layers.dense(inputs=net, units=latent_dim, activation=None)
        ##### detailed structure#########
        #W_enc = weight_variables([n_pixels, h_dim], "W_enc")
        #b_enc = bias_variable([h_dim], "b_enc")
        ## tanh - activation function        avoid vanishing gradient in generative models
        #h_enc = tf.nn.tanh(FC_Layer(X, W_enc, b_enc))

        # layer 2   Output mean and std of the latent variable distribution
        #W_mu = weight_variables([h_dim, latent_dim], "W_mu")
        #b_mu = bias_variable([latent_dim], "b_mu")
        #mu = FC_Layer(h_enc, W_mu, b_mu)
        
        #W_sigma = weight_variables([h_dim, latent_dim], "W_sigma")
        #b_sigma = bias_variable([latent_dim], "b_sigma")
        #sigma = FC_Layer(h_enc, W_sigma, b_sigma)


        # Reparameterize import Randomness
        noise = tf.random_normal([1, latent_dim])
        # z is the ultimate output(latent variable) of our Encoder
        z = mu + tf.multiply(noise, tf.exp(0.5*sigma))
        return mu, sigma, z

############################ Dencoder ############################
def decoder(z):
    '''Z: random_input 1d(latent_dim,)'''
    # layer 1
    with tf.name_scope('Decoder'):
        net = tf.layers.dense(inputs=z, units=h_dim, activation=tf.nn.relu)
        ##### high-way net
        #H = tf.layers.dense(net, units=128, activation=tf.nn.relu, name="deH1")
        #T = tf.layers.dense(net, units=128, activation=tf.nn.sigmoid, name="deT1")
        #C = 1. - T
        #net = H * T + net * C
        #####
        #net = tf.contrib.layers.fully_connected(net, 300, activation_fn=tf.nn.tanh)
        ##### high-way net
        #H = tf.layers.dense(net, units=300, activation=tf.nn.relu, name="deH2")
        #T = tf.layers.dense(net, units=300, activation=tf.nn.sigmoid, name="deT2")
        #C = 1. - T
        #net = H * T + net * C
        ####
        reconstruction = tf.layers.dense(inputs=net, units=n_pixels, activation=tf.nn.sigmoid)
        #W_dec = weight_variables([latent_dim, h_dim], "W_dec")
        #b_dec = bias_variable([h_dim], "b_dec")
        ## tanh - decode the latent representation
        #h_dec = tf.nn.tanh(FC_Layer(z, W_dec, b_dec))

        # layer2 - reconstruction the image and output 0 or 1
        #W_rec = weight_variables([h_dim, n_pixels], "W_dec")
        #b_rec = bias_variable([n_pixels], "b_rec")
        ## 784 bernoulli parameter Output
        #reconstruction = tf.nn.sigmoid(FC_Layer(h_dec, W_rec, b_rec))
        return reconstruction

############################# COnv Encoder ############################
#def encoder(X, num_filters=[32, 64, 64]):
    #'''parameters from
    #https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
    #Conv encoder
    #Param:
        #X:  A batch of real data (MNIST digits).
        #latent_dim: The latent dimensionality.
        #hidden_size: The size of the neural net hidden layers.
    #Returns:
        #mu: Mean parameters for the variational family Normal
        #sigma: Standard deviation parameters for the variational family Normal
    #often in convolution padding = 'same', in max_pooling padding = 'valid'''
    #with tf.name_scope("enc_conv"):
        #net = tf.reshape(X, [-1, height, width, 1])
        #net = tf.layers.conv2d(
                                                #inputs = net,
                                                #filters = 32,
                                                #kernel_size = [3, 3],
                                                #padding = 'same',
                                                #activation=tf.nn.relu)
        #print net.shape
        #net = tf.layers.conv2d(
                                                #inputs = net,
                                                #filters = 64,
                                                #kernel_size = [3, 3],
                                                #padding = 'same',
                                                #strides = (2, 2),
                                                #activation=tf.nn.relu)
        #print net.shape
        #net = tf.layers.conv2d(
                                                #inputs = net,
                                                #filters = 64,
                                                #kernel_size = [3, 3],
                                                #padding = 'same',
                                                #activation=tf.nn.relu)
        ##print net.shape
        ##net = tf.layers.conv2d(
                                                ##inputs = net,
                                                ##filters = 64,
                                                ##kernel_size = [3, 3],
                                                ##padding = 'same',
                                                ##activation=tf.nn.relu)
        #print net.shape
    #with tf.name_scope("dense_enc"):
        #shape_b4_flatten = net.get_shape().as_list()[1:]
        #print "shape_b4_flatten", shape_b4_flatten
        #net = tf.layers.flatten(net)
        #print net.shape
        #net = tf.layers.dense(inputs=net, units=32, activation=tf.nn.relu)
        #print net.shape
        #mu = tf.layers.dense(inputs=net, units=latent_dim, activation=None)
        #sigma = tf.layers.dense(inputs=net, units=latent_dim, activation=None)
         ## Reparameterize import Randomness
        #noise = tf.random_normal([1, latent_dim])
        ## z is the ultimate output(latent variable) of our Encoder
        #z = mu + tf.multiply(noise, tf.exp(0.5*sigma))
        #return mu, sigma, z, shape_b4_flatten
        
############################# Conv Dencoder ############################
#def decoder(z, shape_b4_flatten ):
    #'''Conv decoder
    #z: [batch, latent_dim]'''
    #with tf.name_scope("conv_decoder"):
        ####
        #print "####### decoder"
        ##ipdb.set_trace()
        #dense_shape = shape_b4_flatten[0]*shape_b4_flatten[1]*shape_b4_flatten[2]
        #net = tf.layers.dense(inputs=z, units=dense_shape, activation=tf.nn.relu)
        #net = tf.reshape(net, [-1, shape_b4_flatten[0], shape_b4_flatten[1], shape_b4_flatten[2]])
        #net = tf.layers.conv2d_transpose(
                                                                   #inputs = net,
                                                                    #filters = 32,
                                                                    #kernel_size = [3, 3],
                                                                    #padding = 'same',
                                                                    #strides = (2, 2),
                                                                    #activation=tf.nn.relu)
        #print net.shape
        #net = tf.layers.conv2d(
                                                #inputs = net,
                                                #filters = 1,
                                                #kernel_size = [3, 3],
                                                #padding = 'same',
                                                #activation = tf.nn.sigmoid)
        #print net.shape         ##########(?, 28, 28, 1)
        #net = tf.layers.flatten(net)
        #return net
        
# Loss function = reconstruction error + regularization(similar image's latent representation close)
def train(X):
    mu, sigma, z = encoder(X)
    reconstruction = decoder(z)
    with tf.name_scope('loss'):
        Log_loss = tf.reduce_sum(X * tf.log(reconstruction + 1e-9) + (1 - X) * tf.log(1 - reconstruction + 1e-9), reduction_indices=1)
        #Log_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(X, reconstruction)) 
        KL_loss = -0.5 * tf.reduce_sum(1 + 2*sigma - tf.pow(mu, 2) - tf.exp(2 * sigma), reduction_indices=1)
        VAE_loss = tf.reduce_mean(Log_loss - KL_loss)
        
    #Outputs a Summary protocol buffer containing a single scalar value.
    tf.summary.scalar('VAE_loss', VAE_loss)
    tf.summary.scalar('KL_loss', tf.reduce_mean(KL_loss))
    tf.summary.scalar('Log_loss',  tf.reduce_mean(Log_loss))
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
    ## Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    #store value for these 3 terms so we can plot them later
    vae_loss_array = []
    log_loss_array = []
    KL_term_array = []
    test_vae_array = []
    #iteration_array = [i*record_every for i in range(num_iterations/record_every)]

    for i in range(num_iterations):
        save_name = results_dir + '/' + "_step{}_".format(i)
        # np.round to make MNIST binary
        #get first batch (200 digits)
        x_batch = np.round(mnist.train.next_batch(batch_size)[0])
        #run our optimizer on our data
        _, summary = sess.run([optimizer, summaries], feed_dict={X: x_batch})
        writer.add_summary(summary, i)
        if i % test_every == 0:
            vae_temp = VAE_loss.eval({X : mnist.test.images[0:200]})
            test_vae_array = np.append(test_vae_array, vae_temp)
            summary = sess.run(test_loss_sum, {test_loss: vae_temp})    ## add test score to summary
            recon_test = sess.run(reconstruction, {X : mnist.test.images[0:10]})    ## add test score to summary
            writer.add_summary(summary, i%test_every)
        if (i % record_every == 0):
            #every 1K iterations record these value
            temp_vae = VAE_loss.eval(feed_dict={X: x_batch})
            temp_log = np.mean(Log_loss.eval(feed_dict={X: x_batch}))
            temp_KL = np.mean(KL_loss.eval(feed_dict={X: x_batch}))
            vae_loss_array.append(temp_vae)
            log_loss_array.append(temp_log)
            KL_term_array.append(temp_KL)
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term{}".format(i, temp_vae, temp_log, temp_KL )

        if (i % save_every == 0):
            saver.save(sess, logdir + '/' + str(i))
            

        if (i % plot_every == 0 and i>plot_every):
            #ipdb.set_trace()
            #plot_prior(model_No)
            plot_test(mnist.test.images[0:10], recon_test, save_name=save_name+'test_recon.png')

            ### prior
            nx = ny = 16
            x_values = np.linspace(0.05, 0.95, nx)
            y_values = np.linspace(0.05, 0.95, ny)
            canvas = np.zeros((height * ny, width * nx))
            z_sample = tf.placeholder(tf.float32)
            for ii, yi in enumerate(x_values):
              for jj, xi in enumerate(y_values):
                latent = np.array([[xi, yi]])  #sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
                x_reconstruction = sess.run(reconstruction, feed_dict={z: latent})
                canvas[ii * height:(ii+1) * height, jj *
                       width:(jj + 1) * width] = x_reconstruction.reshape(height, width)
            plt.imshow(canvas)
            plt.savefig(save_name+"_prior.png", format="png")
            plt.close()

            ########## cluster the latent
            # a 2d plot of 10 digit classes in latent space
            x_test, y_test = mnist.test.next_batch(5000)
            x_test_encoded=z.eval({X: x_test})
            plt.figure(figsize=(6,6))
            plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test.argmax(1))
            plt.colorbar()
            plt.savefig(save_name + 'latent cluster.png', format='png')
            plt.close()
            ########## plot the losses
            plt.figure()
            plt.plot(np.arange(len(vae_loss_array)), vae_loss_array, color = 'darkmagenta')
            plt.plot(np.arange(len(vae_loss_array)),  KL_term_array, color = 'slateblue')
            plt.plot(np.arange(len(vae_loss_array)),  log_loss_array, color = 'darkcyan')
            plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], loc='best')
            plt.xlabel("training batches/100")
            plt.title('Loss per iteration')
            plt.savefig(save_name+"_iter{}_loss.png".format(i), format="png")
            plt.close()

if __name__ == "__main__":
    train(X)


'''
Iteration: 1000, Loss: -765.553222656, log_loss: -661.36529541, KL_term91.2270965576
Iteration: 10000, Loss: -538.278320312, log_loss: -509.054290771, KL_term29.9178237915
Iteration: 20000, Loss: -389.014526367, log_loss: -314.975982666, KL_term49.5837974548
Iteration: 40000, Loss: -255.193435669, log_loss: -222.865005493, KL_term37.49036026
Iteration: 69000, Loss: -229.900802612, log_loss: -201.257736206, KL_term26.7798843384
Iteration: 99500, Loss: -209.463882446, log_loss: -186.51159668, KL_term24.2162322998
Iteration: 100000, Loss: -209.921600342, log_loss: -184.041793823, KL_term24.8351421356
'''
'''
Tight structure
Iteration: 0, Loss: -545.54510498, log_loss: -544.708496094, KL_term1.22323262691
Iteration: 1000, Loss: -543.84564209, log_loss: -543.168212891, KL_term0.70369374752
Iteration: 10000, Loss: -460.915893555, log_loss: -440.416870117, KL_term11.686624527
Iteration: 20000, Loss: -282.152435303, log_loss: -250.388717651, KL_term29.1655445099
Iteration: 30000, Loss: -240.628707886, log_loss: -224.83052063, KL_term21.1300430298
Iteration: 240000, Loss: -200.894180298, log_loss: -191.924041748, KL_term7.26480531693
Iteration: 250000, Loss: -198.974182129, log_loss: -191.537963867, KL_term6.95165205002

'''

'''
Tight_resi
Iteration: 0, Loss: -543.70501709, log_loss: -543.341003418, KL_term0.119708031416
Iteration: 1000, Loss: -543.204040527, log_loss: -543.199584961, KL_term0.086833640933
Iteration: 10000, Loss: -418.716339111, log_loss: -437.603118896, KL_term13.6256456375
Iteration: 12000, Loss: -253.138717651, log_loss: -243.541442871, KL_term17.1780071259
Iteration: 30000, Loss: -208.114212036, log_loss: -204.98387146, KL_term5.72965431213
Iteration: 40000, Loss: -219.733352661, log_loss: -204.187011719, KL_term4.81987428665
Iteration: 50000, Loss: -203.907104492, log_loss: -198.388320923, KL_term4.72220468521
Iteration: 60000, Loss: -202.646682739, log_loss: -196.731491089, KL_term4.66666889191

'''
