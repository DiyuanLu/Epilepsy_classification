import tensorflow as tf
import numpy as np
import os
import ipdb



##########################
def network(x):
    '''Get x as input and through whole lot of layers to get an output prediction'''
    ## CNN  (input image-like data; output labels/ image-like)
    ## RNN (input sequence data; output labels/ sequence)
    ## LSTM (input sequence data; output labels/ sequence)
    ## Dilated layers
    ## Residual connection
    ## High-way net(residual block)
    ## variational autoencoder (input any-data; output latent representation)
    ## GAN  (input real-data + random_noise; output real-data like)
    ## WaveNet (input sequnce data.)
    
#################################
def train():
    with tf.name_scope("input"):   #### input data placeholder
        #x_data = tf.placeholder(tf.float32, [None, in_dim])
        #y_data = tf.placeholder(tf.float32, [None, y_dim])
        x_data = 

    output = network(x_data)
        
    with tf.name_scope('loss'):
        ### possible losses in tf.losses.
        loss = tf.losses.mean_squared_error(labels, predictions)
        optimizor = tf.train.AdamOptimizer().minimize(loss)
    tf.summary.scalar('loss', loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()

    saver = tf.train.Saver()
    for step in range(total_steps):
        x_data = load_xdata()
        y_data = load_ydata()
        summary, _ = sess.run([summaries, optimizer])

        if (i % save_every == 0):
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            saver.save(sess, logdir + '/' + str(i))
