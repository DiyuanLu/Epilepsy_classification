#!/usr/bin/python
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import tensorflow as tf
import numpy as np
#import matplotlib
#matplotlib.use('Agg') 
from tensorflow.examples.tutorials.mnist import input_data
from optparse import OptionParser
import matplotlib.pyplot as plt
#from utils import get_variables, linear, AdamOptimizer, CMajorScaleDistribution
import sys
import random
import ipdb
import pandas as pd
import os
import datetime

import argparse
"""
Authors:    Dario Cazzani
https://towardsdatascience.com/generating-digits-and-sounds-with-artificial-neural-nets-ca1270d8445f
"""

'''python3 vae_denoising.py --restore_dir=results/cpu-batch64/2018-07-03T12-00-41-training_20_perfect/model/
'''
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='EEG VAE')

    parser.add_argument("--TRAIN_AUDIO",  default=True,
                          action="store_true",help="Train on audio - otherwise train on MNIST")
    

    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory '
                        'in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--logdir_root', type=str, default='/results',
                        help='Root directory to place the logging '
                        'output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. '
                        'Cannot use with --logdir.')
    return parser.parse_args()


def save_model(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir))
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')
    

def load(saver, sess, logdir):
    #print("Trying to restore saved checkpoints from {} ...".format(logdir),
          #end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('ch')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None

def get_variables(shape, scope):
    xavier = tf.contrib.layers.xavier_initializer()
    const = tf.constant_initializer(0.1)
    W = tf.get_variable('weight_{}'.format(scope), shape, initializer=xavier)
    b = tf.get_variable('bias_{}'.format(scope), shape[-1], initializer=const)
    return W, b

def linear(_input, output_dim, scope=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = [int(_input.get_shape()[1]), output_dim]
        W, b = get_variables(shape, scope)
        return tf.matmul(_input, W) + b

def AdamOptimizer(loss, lr, beta1, var_list=None, clip_grad=False):
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
    if not var_list:
        grads_and_vars = optimizer.compute_gradients(loss)
    else:
        grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    if clip_grad:
        grads_and_vars = [(tf.clip_by_norm(grad, 1), var) for grad, var in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op, grads_and_vars

def CMajorScaleDistribution(num_samples, batch_size):
    sample_rate = 16000
    seconds = 2
    t = np.linspace(0, seconds, sample_rate*seconds + 1)
    C_major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
    while True:
        try:
            batch_x = []
            batch_y = []
            for i in range(batch_size):
                # select random note
                note = C_major_scale[np.random.randint(len(C_major_scale))]
                sound = np.sin(2*np.pi*t*note)
                noise = [random.gauss(0.0, 1.0) for i in range(sample_rate*seconds + 1)]
                noisy_sound = sound + 0.08 * np.asarray(noise)
                start = np.random.randint(0, len(noisy_sound)-num_samples)   ##randomly get num_samples, doen't mater where it starts
                end = start + num_samples
                batch_x.append(noisy_sound[start:end])   ## noisy input
                batch_y.append(sound)       ### ground truth

            yield np.asarray(batch_x), np.asarray(batch_y)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit(1)


def find_files(directory, pattern='Data*.csv', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    import fnmatch
    
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'Data_F' in filename:
                    label = '1'
                elif 'Data_N' in filename:
                    label = '0'
                #if 'base' in filename:
                    #label = '0'
                #elif 'tip' in filename:
                    #label = '1'
                #elif 'seizure' in filename:
                    #label = '2'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    random.shuffle(files)   # randomly shuffle the files
    return files

def read_data(filename, header=None, ifnorm=True ):
    '''read data from .csv
    return:
        data: 2d array [seq_len, channel]'''
    from scipy.stats import zscore
    
    data_temp = pd.read_csv(filename, header=header, nrows=None)
    data_temp = data_temp.values   ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data_temp)
        #data = data_norm    ##data.shape (2048, 1)
        data_norm = (data_norm - np.min(data_norm)) / np.maximum(np.max(data_norm) - np.min(data_norm), 1e-5)
        data = data_norm    ##data.shape (2048, 1)
        #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
        assert np.min(data_norm) >= 0
    return data

def train_test_split_my_data(data_dir, pattern='Data*.csv', num_samples=784, withlabel=False):
    '''load data from directory, split into train and test
    param:
        data_dir
    return:
        datas_train: num_samples*(seq_len*2)
        datas_test: num_samples*(seq_len*2)'''
    from sklearn.model_selection import train_test_split
    files = find_files(data_dir, pattern='Data*.csv', withlabel=False)
    
    datas = np.zeros((len(files), 10240*2))   ## flatten the 2 channels data
    #ipdb.set_trace()
    for ind, filename in enumerate(files):
        if ind % 1000 == 0:
            print("file", ind)
        data = read_data(filename, header=None, ifnorm=True)
        datas[ind, :] = np.append(data[:, 0], data[:, 1])  ## only use one channel

    train, test = train_test_split(datas, test_size=0.2, random_state=999)
    
    len_use = (train.size // input_dim) * input_dim
    datas_train = train.flatten()[0: len_use].reshape(-1, input_dim)
    len_use = (test.size // input_dim) * input_dim
    datas_test = test.flatten()[0: len_use].reshape(-1, input_dim)
    
    return datas_train, datas_test, datas_train.shape[0], datas_test.shape[0]


def EEG_data(data, pattern='Data*.csv', withlabel=False, num_samples=784, batch_size=128):
    '''load train_data, segment into chuncks of input_dim long
    using generator yield batch
    '''
    while True:
        try:
            for ii in range(len(data)//batch_size):
                batch_x = data[ii*batch_size: (ii+1)*batch_size, :]

                yield np.asarray(batch_x)

        except Exception as e:
            print('Could not produce batch of sinusoids because: {}'.format(e))
            sys.exit(1)


def lr(epoch):
    learning_rate = 0.0025
    #if epoch > 400:
        #learning_rate *= 0.5e-3
    #elif epoch > 350:
        #learning_rate *= 1e-3
    #elif epoch > 200:
        #learning_rate *= 1e-2
    #elif epoch > 100:
        #learning_rate *= 1e-1
    return learning_rate

# Get the MNIST data
#mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

# Parameters
input_dim = 128#mnist.train.images.shape[1]   perfect-2048
hidden_layer1 = input_dim // 4   ## best result config   perfect-1024
hidden_layer2 = hidden_layer1 // 2   ###                      perfect-256
z_dim = hidden_layer2 // 2    ###perfect-128

beta1 = 0.9
batch_size = 2
epochs = 501
tensorboard_path = 'tensorboard_plots/'
noise_length = int(input_dim / 5.)
data_dir = 'data'     ##'../data/train_data'  ##
pattern='Data*.csv'
version = 'progress_{}'.format(pattern[0:4])
# get audio data
#audio_data = CMajorScaleDistribution(input_dim, batch_size)
#train_data, test_data, num_train, num_test = train_test_split_my_data(data_dir, pattern='Data*.csv', withlabel=False)
#num_train = train_data.shape[0]
#num_test = test_data.shape[0]
##ipdb.set_trace()
#train_batch = EEG_data(train_data, pattern=pattern, withlabel=False, num_samples=input_dim, batch_size=batch_size)
#test_data = get_test_data(datas_test, num_samples=input_dim)
learning_rate = tf.placeholder("float32")

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir= 'results/cpu-batch{}/load_add20--lr{}-hid{}-{}-z{}-input_dim{}-'.format(batch_size, 0.0001, hidden_layer1, hidden_layer2, z_dim, input_dim) + datetime + '/'#cnv4_lstm64test
logdir = results_dir+ "model/"


# p(z|X)
def encoder(x):
    with tf.variable_scope("encoder") as scope:
    #e_linear_1 = tf.nn.relu(linear(x, hidden_layer1, 'e_linear_1'))
    #e_linear_2 = tf.nn.relu(linear(e_linear_1, hidden_layer2, 'e_linear_2'))
        e_linear_1 = tf.layers.dense(inputs=x, units=hidden_layer1, activation=tf.nn.leaky_relu, name='e_linear_1')
        e_linear_1 = tf.layers.dropout(e_linear_1, rate=0.5)
        e_linear_2 = tf.layers.dense(inputs=e_linear_1, units=hidden_layer2, activation=tf.nn.leaky_relu, name='e_linear_2')
        e_linear_2 = tf.layers.dropout(e_linear_2, rate=0.5)
        z_mu = tf.layers.dense(inputs=e_linear_2, units=z_dim, activation=None, name='z_mu')
        z_logvar = tf.layers.dense(inputs=e_linear_2, units=z_dim, activation=None, name='z_logvar')
    return z_mu, z_logvar

    
#def encoder(x, num_filters=[4, 8, 16], kernel_size=[4, 1], pool_size=[4, 1], height=2048, width=1, scope=None):
    #"""parameters from
    #https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder
    #def encoder_net(x, latent_dim, h_dim):
    #Construct an inference network parametrizing a Gaussian.
    #Args:
    #x: A batch of real data (MNIST digits).
    #latent_dim: The latent dimensionality.
    #hidden_size: The size of the neural net hidden layers.
    #Returns:
    #mu: Mean parameters for the variational family Normal
    #sigma: Standard deviation parameters for the variational family Normal
    #often in convolution padding = 'same', in max_pooling padding = 'valid'
    #"""

    #with tf.variable_scope(scope, 'encoder'):
        #net = tf.reshape(x, [-1,  height, width, 1])

    ## Convolutional Layer 
        #for layer_id, num_outputs in enumerate(num_filters):   ## avoid the code repetation
            #with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                #net = tf.layers.conv2d(
                                        #inputs = net,
                                        #filters = num_outputs,
                                        #kernel_size = kernel_size,
                                        #strides = (4, 1),
                                        #padding='same',
                                        #activation=tf.nn.leaky_relu)
                ##net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, padding='SAME', strides=2)
                #print("net ", net.shape.as_list())
                #net = tf.layers.batch_normalization(net)
        #### dense layer
        #with tf.name_scope("dense"):
            
            #net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])
            #net = tf.layers.dense(inputs=net, units=hidden_layer2, activation=tf.nn.relu)
            #net = tf.layers.dropout(inputs=net, rate=0.5)
        #print("dense out net ", net.shape.as_list())
        #### Get
        #z_mu = tf.layers.dense(inputs=net, units=z_dim, activation=None, name='z_mu')
        #z_logvar = tf.layers.dense(inputs=net, units=z_dim, activation=None, name='z_logvar')

        #return z_mu, z_logvar    #dense1

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

# p(X|z)
def decoder(z):
    with tf.variable_scope("decoder") as scope:
        #d_linear_1 = tf.nn.relu(linear(z, hidden_layer2, 'd_linear_1'))
        #d_linear_2 = tf.nn.relu(linear(d_linear_1, hidden_layer1, 'd_linear_2'))
        d_linear_1 = tf.layers.dense(inputs=z, units=hidden_layer2, activation=tf.nn.leaky_relu, name='d_linear_1')
        d_linear_1 = tf.layers.dropout(d_linear_1, rate=0.5)
        d_linear_2 = tf.layers.dense(inputs=d_linear_1, units=hidden_layer1, activation=tf.nn.leaky_relu, name='d_linear_2')
        d_linear_2 = tf.layers.dropout(d_linear_2, rate=0.5)
        logits = tf.layers.dense(inputs=d_linear_2, units=input_dim, activation=None)
        prob = tf.nn.sigmoid(logits)
    return prob, logits


def upsample(inputs, name='depool', factor=[2,1]):
    size = [int(inputs.shape[1] * factor[0]), int(inputs.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(inputs, size=size, align_corners=None, name=None)
    return out

    
#def decoder(z, num_filters=[8, 1], kernel_size=5, height=2048, width=1, scope=None):
    #"""Build a generative network parametrizing the likelihood of the data
    #Args:
    #inputs_dec: Samples of latent variables with size latent_dim_2
    #hidden_size: Size of the hidden state of the neural net
    #Returns:
    #reconstruction: logits for the Bernoulli likelihood of the data
    #"""
    #net = z
    #print net.shape
    #with tf.variable_scope(scope, 'dec'):
        #with tf.name_scope('dec_fc_dropout'):
            #net = tf.layers.dense(inputs=net, units=hidden_layer2, activation=tf.nn.relu)
            #net = tf.layers.dropout(inputs=net, rate=0.5, name='dec_dropout1')
            #net = tf.layers.dense(inputs=net, units=input_dim, activation=tf.nn.relu)
            #net = tf.layers.dropout(inputs=net, rate=0.5, name='dec_dropout2')
            ##net = tf.reshape(net, [-1, 1024, 1, 1])
            ##print("decoder dense net ", net.shape.as_list())
            ############# deconvolution layer
            
            ##net = upsample(net)
            ##print("upsample net ", net.shape.as_list())
            ##for layer_id, num_outputs in enumerate(num_filters):
                ##with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
                    ##net = tf.layers.conv2d(
                                            ##inputs = net,
                                            ##filters = num_outputs,
                                            ##kernel_size = kernel_size,
                                            ##padding='SAME',
                                            ##activation=tf.nn.sigmoid)
                    ##tf.summary.histogram('activation', net)
                    ##print("decoder net ", net.shape.as_list())
            ##shape = net.get_shape().as_list()
            ##assert len(shape) == len(output_dim), 'shape mismatch'
            ##### reconstruction activ = sigmoid
            #reconstruction  = tf.reshape(net, [-1, height * width])
            #print("reconstruction", reconstruction.shape.as_list())
            #return reconstruction
            
def train():

    args = get_arguments()
    if not args.restore_from:
        restore_from = logdir
    else:
        restore_from = args.restore_from

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with tf.name_scope('Input'):
        X = tf.placeholder(tf.float32, shape=[None, input_dim])
        X_noisy = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='Input_noisy')   ## input original
        #input_images = tf.reshape(X_noisy, [-1, 28, 28, 1])
        if args.TRAIN_AUDIO:
            ## Audio inputs normalization
            ##X_norm = tf.div(tf.add(X, 1.), 2)
            #X_norm = tf.div(tf.add(X, -1.0*np.min(X)), 2)   ### normalize by the absolute min values
            X_norm = X   ### normalize by the absolute min values
        train_data, test_data, num_train, num_test = train_test_split_my_data(data_dir, pattern='Data*.csv', withlabel=False)
        
        num_train = train_data.shape[0]
        num_test = test_data.shape[0]
        #ipdb.set_trace()
        train_batch = EEG_data(train_data, pattern=pattern, withlabel=False, num_samples=input_dim, batch_size=batch_size)

    ## restore model
    with tf.name_scope('Latent_variable'):
        z = tf.placeholder(tf.float32, shape=[None, z_dim])

    with tf.variable_scope('Encoder'):
        z_mu, z_logvar = encoder(X)##, height=input_dim, width=1

    with tf.variable_scope('Decoder') as scope:
        z_sample = sample_z(z_mu, z_logvar)
        #ipdb.set_trace()
        decoder_output, logits = decoder(z_sample)
        tf.summary.histogram("decode", decoder_output)

        
        if args.TRAIN_AUDIO:
            # audio input "de-normalization"
            #decoder_output = tf.subtract(tf.multiply(decoder_output, 2.), 1.)
            decoder_output = decoder_output
            
        #generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])
        # Unused, unless you want to generate new data from N~(0, 1)
        scope.reuse_variables()
        ## Sampling from random z
        X_samples, _ = decoder(z)

    with tf.name_scope('Loss'):
        if args.TRAIN_AUDIO:
            reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X_norm), 1)
        else:
            reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
        # VAE loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    # Optimizer
    train_op, grads_and_vars = AdamOptimizer(vae_loss, learning_rate, beta1)

    # Visualization
    tf.summary.scalar(name='Loss', tensor=vae_loss)
    tf.summary.histogram(name='Sampled variable', values=z_sample)
    saver = tf.train.Saver(max_to_keep= 20)
    #for grad, var in grads_and_vars:
        #tf.summary.histogram('Gradients/' + var.name, grad)
        #tf.summary.histogram('Values/' + var.name, var)

    #if options.TRAIN_AUDIO:
        #tf.summary.audio(name='Input Sounds', tensor=X_noisy, sample_rate = 16000, max_outputs=3)
        #tf.summary.audio(name='Generated Sounds', tensor=decoder_output_denorm, sample_rate = 16000, max_outputs=3)
    #else:
        #tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
        #tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)

    summary_op = tf.summary.merge_all()

    step = 0
    init = tf.global_variables_initializer()
    n_batches = num_train // batch_size
    print("n_batches", n_batches)
    #n_batches = 260
    with tf.Session() as sess:
        sess.run(init)
        try:
            saved_global_step = load(saver, sess, restore_from)
            if is_overwritten_training or saved_global_step is None:
                # The first training step will be saved_global_step + 1,
                # therefore we put -1 here for new or overwritten trainings.
                saved_global_step = -1

        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.")
            raise

        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        print(results_dir)
        try:
            if args.TRAIN_AUDIO:
                writer = tf.summary.FileWriter(logdir=tensorboard_path+'/audio/', graph=sess.graph)
            else:
                writer = tf.summary.FileWriter(logdir=tensorboard_path+'/mnist/', graph=sess.graph)
            loss_epoch = []
            loss_test = []
            
            for epoch in range(epochs):
                count = 0
                batch_loss_tot = 0
                for iteration in range(n_batches):
                    if args.TRAIN_AUDIO:
                        batch_x = train_batch.next()   ### python2.7 batch_size*input_dim
                        #batch_x = train_batch.__next__()   ### python3 batch_size*input_dim
                        
                    else:
                        batch_x, _ = mnist.train.next_batch(batch_size)

                    noisy_batch = batch_x 
                    # Train
                    batch_x = np.reshape(batch_x, [-1, input_dim])
                    sess.run(train_op, feed_dict={X: batch_x, X_noisy: noisy_batch, learning_rate: lr(epoch)})                                           
                    
                    
                    if iteration % 20 == 0:
                        summary, _, batch_loss = sess.run([summary_op, train_op, vae_loss], feed_dict={X: batch_x, X_noisy: noisy_batch, learning_rate: lr(epoch)})
                        batch_loss_tot += batch_loss
                        count += 1                   
                    
                loss_epoch.append(batch_loss_tot / (count) )
                    
                if epoch % 1 == 0:
                    #ipdb.set_trace()
                    rand_ind = np.random.choice(num_test, 81)
                    examples_test = test_data[rand_ind, :]
                    test_data = np.reshape(test_data, [-1, input_dim])
                    examples_test = np.reshape(examples_test, [-1, input_dim])
                    loss, recon = sess.run([vae_loss, decoder_output], feed_dict={X: test_data, X_noisy: test_data})
                    recon = sess.run( decoder_output, feed_dict={X: examples_test, X_noisy: examples_test})
                    loss_test.append(loss)
                    print("Epoch: {} - iteration {} - TrainLoss: {:.4f} - TestLoss: {:.4f}\n".format(epoch, iteration, batch_loss, loss))

                if epoch % 50 == 0:
                    
                    fig, axs = plt.subplots(9, 9, figsize=(20,10))  ##, subplot_kw={'xticks': []}
                    #fig.set_title("Reconstructed samples")
                    for ind, ax in enumerate(axs.flat):
                        #ax.plot(np.arange(input_dim)/512.0, recon[ind, :], 'c', label='recon')
                        #ax.plot(np.arange(input_dim)/512.0, examples_test[ind, :], 'm', label='original')
                        ax.plot(np.arange(input_dim) / 512.0, recon[ind, :], 'c', label='recon')
                        ax.plot(np.arange(input_dim) / 512.0, examples_test[ind, :], 'm', label='original')
                        #plt.setp(ax.get_xticklabels(), visible = False)
                        plt.setp(ax.get_yticklabels(), visible = False)
                    #ipdb.set_trace()
                    axs.flat[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                    fig.text(0.5, 0.05, 'time / s', fontsize=22)
                    fig.text(0.05, 0.5, 'Normalized amplitude', fontsize=22)
                    plt.savefig(results_dir +'Epoch_{}_recon_ori_test.png'.format(epoch), format='png')
                    plt.close()

                    plt.figure()
                    plt.plot(np.array(loss_epoch), color='skyblue', label='train')
                    plt.plot(np.array(loss_test), color='violet', label='test')
                    plt.legend()
                    plt.title("Loss")
                    plt.xlabel("epochs")
                    plt.ylabel("loss")
                    plt.savefig(results_dir +'Epoch_{}_loss.png'.format(epoch), format='png')
                    plt.close()

                    #ipdb.set_trace()
                    np.savetxt(results_dir + '/' +'batch_accuracy_per_class.csv', (np.array(loss_epoch),np.array(loss_test) ), header='loss_train,loss_test', delimiter=',', fmt="%10.5f", comments='')

                if epoch % 50 == 0:
                    save_model(saver, sess, logdir, epoch)
                    last_saved_step = epoch
                    
            writer.add_summary(summary, global_step=iteration)
            print("Model Trained!")

        except KeyboardInterrupt:
            print('Stopping training...')

if __name__ == '__main__':
    
    train()
