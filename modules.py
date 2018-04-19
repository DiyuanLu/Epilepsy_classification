import numpy as np
import tensorflow as tf
import ipdb

num_classes = 2


def dense_net(x):
    '''with dense and dropout
    x: 2d array Batch_size * samples'''
    dense1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5)
    dense2 = tf.layers.dense(inputs=dropout1, units= 512, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5)
    dense3 = tf.layers.dense(inputs=x, units=64, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.5)

    return dropout
    
def fc_net(x, hid_dims=[500, 300, 100]):
    net = x
    # Convolutional Layer 
    for layer_id, num_outputs in enumerate(hid_dims):   ## avoid the code repetation
        with tf.variable_scope('fc_{}'.format(layer_id)) as layer_scope:
            net = tf.contrib.layers.fully_connected(
                                                                    net,
                                                                    num_outputs,
                                                                    activation_fn=tf.nn.relu)
            net = tf.contrib.layers.batch_norm(
                                                                net,
                                                                center = True,
                                                                scale = True)
        with tf.variable_scope('fc_out'):
            net = tf.contrib.layers.fully_connected(
                                                                    net,
                                                                    num_classes,
                                                                    activation_fn=tf.nn.sigmoid)
            tf.summary.histogram('activation', net)
            return net
           

def resi_net(x, hid_dims=[500, 300]):
    '''tight structure of fully connected and residual connection
    x: [None, seq_len, width]'''
    net = tf.layers.flatten(x)
    print "flatten net", net.shape
    for layer_id, num_outputs in enumerate(hid_dims):
        with tf.variable_scope('FcResi_{}'.format(layer_id)) as layer_scope:
            ### layer block #1 fully + high-way + batch_norm
            net = tf.contrib.layers.fully_connected(
                                                                    net,
                                                                    num_outputs,
                                                                    activation_fn=tf.nn.relu)
            #### high-way net
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH1")
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT1")
            C = 1. - T
            net = H * T + net * C
            ### batch normalization
            net = tf.contrib.layers.batch_norm(
                                                                net,
                                                                center = True,
                                                                scale = True)
            with tf.variable_scope('fc_out'):
                ### another fully conn
                outputs = tf.contrib.layers.fully_connected(
                                                                            net,
                                                                            num_classes,
                                                                            activation_fn=tf.nn.sigmoid)
        return outputs
        
def CNN(x, num_filters=[16, 32, 64], seq_len=10240, width=1):
    '''Perform convolution on 1d data'''
    '''Perform convolution on 1d data'''
    ## Input layer
    seq_len = seq_len
    inputs = tf.reshape(x, [-1, seq_len, width, 1])   
    net = inputs
    # Convolutional Layer 
    for layer_id, num_outputs in enumerate(num_filters):   ## avoid the code repetation
        with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
            net = tf.layers.conv2d(
                                                inputs = net,
                                                filters = num_outputs,
                                                kernel_size = [5, 1],
                                                padding = 'same',
                                                activation=tf.nn.relu)
            print net.shape
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
            print net.shape
    ### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])
    net = tf.layers.dense(inputs=net, units=500, activation=tf.nn.relu)
    #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits


def DeepConvLSTM(x, num_filters=[64, 64], filter_size=5, num_lstm=128, seq_len=1280, width=2):
    '''work is inspired by
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    in-shape: (BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)
    another from https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    '''
    net = tf.reshape(x,  [-1, seq_len,  width,1]) 
    print net.shape
    for layer_id, num_outputs in enumerate(num_filters):
        with tf.variable_scope("block_{}".format(layer_id)) as layer_scope:
            net = tf.layers.conv2d(
                                                inputs = net,
                                                filters = num_outputs,
                                                kernel_size = [filter_size, 1],
                                                padding = 'same',
                                                activation=tf.nn.relu
                                                )
            print 'conv{}'.format(layer_id), net.shape
    #ipdb.set_trace()
    with tf.variable_scope("reshape4rnn") as layer_scope:
        ### prepare input data for rnn requirements. current shape=[None, seq_len, num_filters]
        ### Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        #ipdb.set_trace()
        net = tf.reshape(net, [-1, seq_len, width*num_filters[-1]])
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        net = tf.unstack(net, seq_len, 1)
        
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm) 
        net, _ = tf.nn.static_rnn(lstm_layer, net, dtype=tf.float32)  ###net shape: [batch_size, max_time, ...]
        #print net.shape
        #lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_lstm)
        #net, _ = tf.nn.dynamic_rnn(lstm_cell_2, net, dtype=tf.float32)
        #print 'lstm output', net.shape
    with tf.variable_scope("dense_out") as layer_scope:
        #net = tf.reshape(net, [-1, num_lstm])
        #print 'reshape', net.shape
        out_weights=tf.Variable(tf.random_normal([num_lstm, num_classes]))
        out_bias=tf.Variable(tf.random_normal([num_classes]))
        net=tf.matmul(net[-1], out_weights) + out_bias
        #net = tf.layers.dense(inputs=net[-1], units=num_classes, activation=tf.nn.sigmoid)
        tf.summary.histogram('activation', net)
        print 'final output', net.shape

    return net


def RNN(x, num_lstm=128, seq_len=1280, width=2):
    '''Use RNN
    x: shape[batch_size,time_steps,n_input]'''
    with tf.variable_scope("rnn_lstm") as layer_scope:
        net = tf.reshape(x, [-1, seq_len, width])
        #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
        net = tf.unstack(net, seq_len, 1)
        ##### defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm,forget_bias=1)
        outputs, _ =tf.nn.static_rnn(lstm_layer, net, dtype="float32")

        out_weights=tf.Variable(tf.random_normal([num_lstm,num_classes]))
        out_bias=tf.Variable(tf.random_normal([num_classes]))
        net=tf.matmul(outputs[-1], out_weights) + out_bias
        #net = tf.layers.dense(inputs=net[ -1], units=num_classes, activation=tf.nn.sOFTMAX)
        tf.summary.histogram('activation', net)
    return net

def Dilated_CNN(x, num_filters=[16, 32, 64], dilation_rate=[1, 2, 4], seq_len=10240):
    '''Perform convolution on 1d data
    x: shape [batch_size, height, width, channels]
    '''
    ## Input layer
    seq_len = seq_len
    inputs = tf.reshape(x,  [-1, seq_len, width])
    net = inputs        
    # Convolutional Layer 
    for layer_id, num_outputs in enumerate(num_filters):   ## avoid the code repetation
        with tf.variable_scope('block_{}'.format(layer_id)) as layer_scope:
            #### dilated layers
            net = tf.nn.atrous_conv2d(   
                                                net,
                                                [1, 5, 1, num_outputs],     # [filter_height, filter_width, inum_channels, out_channels]
                                                'SAME',
                                                activation=tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
    ### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
    net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits
