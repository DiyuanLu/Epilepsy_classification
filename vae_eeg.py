

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

version = "vae_CNN_MNIST"     ###"VAE_ds16"  #
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
data_dir = "data/train_data"
data_dir_test = "data/test_data"
ratio = 2
save_every = 25 * ratio
plot_every = 20 * ratio
test_every = 5 * ratio
num_iterations = 100 * ratio + 1   # 50
recording_interval = 1000    # 1000   #
print_result = 10 * ratio
pattern = 'ds_8*.csv'
### Hyperparams
#seq_len = 1280 #   640     #10240    #28 * 28   seq=width
#width = seq_len
height, width = 28, 28
hid_dim1 = 500   # Encoder: input -- hidden1 -- latent1 -- hidden2 -- latent2
hid_dim2 = 200
latent_dim = 2
batch_size = 20
epochs = 50
total_batches =  epochs * 3000 // batch_size + 1
results_dir = "results/" + version + '/' + datetime +'bs_' +np.str(batch_size)
logdir = results_dir  + "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir ):
    os.makedirs(results_dir )
print results_dir

def get_test_data():
    test_data = np.empty([0, seq_len])
    test_labels = np.empty([0])
    for filen in files_test:
        data = np.average(func.read_data(filen[0]), axis=0)
        test_data = np.vstack((test_data, data))
        test_labels = np.append(test_labels, filen[1])
    test_labels = np.eye((n_classes))[test_labels.astype(int)]
    return test_data, test_labels


def plot_prior(sess, load_model=False, save_name='save_name'):
    #if load_model:
        #saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    nx = ny = 4
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)
    canvas = np.empty((height * ny, width * nx))
    noise = tf.random_normal([1, latent_dim])
    z = tf.placeholder(tf.float32, [1, latent_dim], 'dec_input')
    reconstruction = decoder(z)
    latent = np.random.randn(1, latent_dim)
    #sess2 = tf.Session()
    #sess2.run(tf.global_variables_initializer())
    for ii, yi in enumerate(x_values):
      for j, xi in enumerate(y_values):
        latent[0, 0:2] = xi, yi  #sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
        x_reconstruction = sess.run(reconstruction, feed_dict={z: latent})
        canvas[(nx - ii - 1) * height:(nx - ii) * height, j *
               width:(j + 1) * width] = x_reconstruction.reshape(height, width)
    plt.savefig(save_name, format="jpg")   # canvas
                        
def upsample(inputs, name='depool', factor=[2,2]):
    size = [int(inputs.shape[1] * factor[0]), int(inputs.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(inputs, size=size, align_corners=None, name=None)
    return out

#def plot_test(original, reconstruction, load_model = False, save_name="save"):
    ## Here, we plot the reconstructed eeg on test set images.
    ##if load_model:
        ##saver.restore(sess, os.path.join(os.getcwd(), logdir + '/' + "{}".format(model_No)))
    #num_pairs = 6
    #original = np.reshape(original, (-1, seq_len))
    #for pair in range(num_pairs):
        ##reshaping to show original test image
        #x_image = np.reshape(original[pair, :], (1,seq_len))
        #index = pair * 2 + 1
        #ax1 = plt.subplot(num_pairs,2,index)  # arrange in 6*2 layout
        #plt.imshow(x_image, aspect="auto")
        #if pair == 0 or pair == 1:
            #plt.title("Original")
        ##plt.xlim([0, 27])
        ##plt.ylim([27, 0])

        #x_reconstruction_image = np.reshape(reconstruction[pair, :], (1,seq_len))
        #ax2 = plt.subplot(num_pairs, 2, (pair + 1) * 2 , sharex = ax1, sharey=ax1)
        #plt.imshow(x_reconstruction_image, aspect="auto")
        #plt.setp(ax2.get_xticklabels(), visible=False)
        ##plt.xlim([0, 27])
        ##plt.ylim([27, 0])
        #plt.tight_layout()
        #if pair == 0 or pair == 1:
            #plt.title("Reconstruct")
            
    #plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.95,
                #wspace=0.30, hspace=0.22)
    #plt.savefig(save_name + "samples.png", format="png")
    #plt.close()

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
   
#### input
with tf.name_scope("input"):
    ## the real data from database
    inputs_enc = tf.placeholder(tf.float32, [None, height * width], name='inputs_enc')


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
def encoder(inputs_enc, num_filters=[32, 64, 64, 64], kernel_size=[3, 3], pool_size=[2, 2], scope=None):
    """parameters from
    https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
    def encoder_net(x, latent_dim, h_dim):
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

    with tf.variable_scope(scope, 'encoder'):
        inputs_enc = tf.reshape(inputs_enc, [-1,  height, width, 1])
        net = inputs_enc
    # Convolutional Layer 
        for layer_id, num_outputs in enumerate(num_filters):   ## avoid the code repetation
            with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                net = tf.layers.conv2d(
                                                    inputs = net,
                                                    filters = num_outputs,
                                                    kernel_size = kernel_size,
                                                    padding='SAME',
                                                    activation=tf.nn.relu)
                #net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, padding='SAME', strides=2)
                print net.shape
                
        ### dense layer
        with tf.name_scope("dense"):
            net = tf.layers.dropout(inputs=net, rate=0.75)
            net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])
            net = tf.layers.dense(inputs=net, units=hid_dim2, activation=tf.nn.relu)
        print net.shape
        ### Get mu
        mu_1 = tf.contrib.layers.fully_connected(net, latent_dim, activation_fn=None)
        # layer 2   Output mean and std of the latent variable distribution
        sigma_1 = tf.contrib.layers.fully_connected(net, latent_dim, activation_fn=None)
        # Reparameterize import Randomness
        noise = tf.random_normal([1, latent_dim])
        # z_1 is the fisrt leverl output(latent variable) of our Encoder
        z_1 = mu_1 + tf.multiply(noise, tf.exp(0.5*sigma_1))
        print z_1.shape
        return mu_1, sigma_1, z_1     #dense1
                                   #

def decoder(inputs_dec, num_filters=[25, 1], kernel_size=5, scope=None):
    """Build a generative network parametrizing the likelihood of the data
    Args:
    inputs_dec: Samples of latent variables with size latent_dim_2
    hidden_size: Size of the hidden state of the neural net
    Returns:
    reconstruction: logits for the Bernoulli likelihood of the data
    """
    net = inputs_dec
    print net.shape
    with tf.variable_scope(scope, 'dec'):
        with tf.name_scope('dec_fc_dropout'):
            net = tf.layers.dense(inputs=net, units=hid_dim2, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=0.75, name='dec_dropout1')
            net = tf.layers.dense(inputs=net, units=14 * 14 * 25, activation=tf.nn.relu)
            net = tf.layers.dropout(inputs=net, rate=0.75, name='dec_dropout2')
            net = tf.reshape(net, [-1, 14, 14, num_filters[0]])
            print net.shape
            ########### deconvolution layer
            net = upsample(net)
            for layer_id, num_outputs in enumerate(num_filters):
                with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                    net = tf.layers.conv2d(
                                                        inputs = net,
                                                        filters = num_outputs,
                                                        kernel_size = kernel_size,
                                                        padding='SAME',
                                                        activation=tf.nn.sigmoid)
                    tf.summary.histogram('activation', net)
                    print net.shape
            #shape = net.get_shape().as_list()
            #assert len(shape) == len(output_dim), 'shape mismatch'
            #### reconstruction activ = sigmoid
            reconstruction  = tf.reshape(net, [-1, height * width])
            
            return reconstruction


def train(input_enc):
    with tf.name_scope("Data"):
        ### Get data
        files_wlabel_train = func.find_files(data_dir, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_wlabel_test = func.find_files(data_dir_test, pattern=pattern, withlabel=True )### traverse all the files in the dir, and divide into batches, from
        files_train, labels_train = np.array(files_wlabel_train)[:, 0], np.array(np.array(files_wlabel_train)[:, 1]).astype(np.int)
        files_test, labels_test = np.array(files_wlabel_test)[:, 0], np.array(files_wlabel_test)[:, 1].astype(np.int)   ## seperate the name and label
        dataset_train = tf.data.Dataset.from_tensor_slices(files_wlabel_train).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_test = tf.data.Dataset.from_tensor_slices(files_wlabel_test).repeat().batch(batch_size).shuffle(buffer_size=10000)
        # create TensorFlow Dataset objects
        dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        dataset_test = tf.data.Dataset.from_tensor_slices((files_test, labels_test)).repeat().batch(batch_size).shuffle(buffer_size=10000)
        iter = dataset_train.make_initializable_iterator()
        iter_test = dataset_test.make_initializable_iterator()
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
    tf.summary.histogram("vae/mu", mu_1)
    tf.summary.histogram("vae/sigma", sigma_1)
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
        filename_train, labels_train =  sess.run(ele)   # names, 1s/0s
        filename_test, labels_test =  sess.run(ele_test)   # names, 1s/0s
        data_train = []
        for ind in range(len(filename_train)):
            data = func.read_data(filename_train[ind], ifaverage=ifaverage, ifnorm=ifnorm)
            data_train.append(data)
        labels_train =  np.eye((num_classes))[labels_train.astype(int)]   # get one-hot lable

        #run our optimizer on our data
        _, summary = sess.run([optimizer, summaries], feed_dict={inputs_enc: data_train})
        writer.add_summary(summary, batch)
        ### test
        if (batch % 10 == 0):
           ##################### test ####################################### 
            data_test = []
            for ind in range(len(filename_test)):
                data = func.read_data(filename_test[ind], ifaverage=ifaverage)
                data_test.append(data)
            labels_test =  np.eye((num_classes))[labels_test.astype(int)]   # get one-hot lable
                
            test_temp = VAE_loss.eval({input_enc : data_test})
            test_vae_array = np.append(test_vae_array, test_temp)
            
            summary = sess.run(test_loss_sum, {test_loss: test_temp})    ## add test score to summary
            writer.add_summary(summary)
            reconstruction_test = reconstruction.eval({input_enc: test_data[0:10]})
            #every 1K iterations record these values
            temp_vae = VAE_loss.eval(feed_dict={inputs_enc: batch_data})
            temp_log = np.mean(Log_loss.eval(feed_dict={inputs_enc: batch_data}))
            temp_KL = np.mean(KL_loss.eval(feed_dict={inputs_enc: batch_data}))
            vae_loss_array.append(temp_vae )
            KL_loss_array.append(temp_KL)
            log_loss_array.append( temp_log)
        if batch % print_result == 0  and batch > 50:
            print "Iteration: {}, Loss: {}, log_loss: {}, KL_term {}".format(batch, temp_vae, temp_log, temp_KL )

        if (batch % save_every == 0  and batch > 50):
            saver.save(sess, logdir + '/' + str(batch) + '_model.ckpt')

        if (batch % plot_every == 0 and batch > 50):
            #ipdb.set_trace()
            plot_test(test_data[0:10], reconstruction_test, save_name=save_name + 'test_reconstruction.png')
            #plot_prior(load_model=False, save_name = save_name + 'prior.png')
            ########### plot prior
            #ipdb.set_trace()
            nx = ny = 4
            x_values = np.linspace(-1, 1, nx)
            y_values = np.linspace(-1, 1, ny)
            canvas = np.zeros((height * ny, width * nx))
            z_sample = tf.placeholder(tf.float32)
            for ii, yi in enumerate(x_values):
              for jj, xi in enumerate(y_values):
                latent = np.array([[xi, yi]])  #sess.run(reconstruction, {z_2: np_z, X: np_x_fixed})
                x_reconstruction = sess.run(reconstruction, feed_dict={z: latent})
                canvas[ii * height:(ii+1) * height, jj *
                       width:(jj + 1) * width] = x_reconstruction.reshape(height, width)
            plt.savefig(save_name+'prior.png', format="jpg")   # canvas

    
            #func.plot_smooth_shadow_curve([np.array(vae_loss_array), np.array(KL_loss_array), np.array(log_loss_array)], ylabel="accuracy", colors=['darkcyan', 'royalblue', 'indigo'], title='Losses during training', labels=['vae_loss', 'KL_loss', 'log_loss'], save_name=results_dir+ "/train_losses_dring_training_batch_{}".format(batch))
            
            #func.plot_smooth_shadow_curve(test_vae_array, colors='c', ylabel="loss", title='Loss in testing',labels='loss_test', save_name=results_dir+ "/test_loss_batch_{}".format(batch))
            ##ipdb.set_trace()
            func.save_data((vae_loss_array, KL_loss_array, log_loss_array), header='accuracy_train,loss_train,accuracy_test,sen_total_train,spe_total_train', save_name=results_dir + '/' +'3losses_class.csv')   ### the header names should be without space! TODO
            
            
            
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
    func.save_data((vae_loss_array, KL_loss_array, log_loss_array), header='vae_loss,KL_loss,log_loss', save_name=results_dir + '/' +'3losses_class.csv')

        
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
