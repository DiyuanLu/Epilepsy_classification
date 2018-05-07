import numpy as np
import tensorflow as tf
import ipdb


def dense_net(x, num_classes = 2):
    '''with dense and dropout
    x: 2d array Batch_size * samples'''
    net = tf.layers.flatten(x)
    net  = tf.layers.dense(inputs=net , units=1024, activation=tf.nn.relu)
    net  = tf.layers.dropout(inputs=net , rate=0.5)
    net  = tf.layers.dense(inputs=net , units= 512, activation=tf.nn.relu)
    net  = tf.layers.dropout(inputs=net , rate=0.5)
    net  = tf.layers.dense(inputs=net , units=64, activation=tf.nn.relu)
    net  = tf.layers.dropout(inputs=net , rate=0.5)

    return net 
    
def fc_net(x, hid_dims=[500, 300, 100], num_classes = 2):
    net = tf.layers.flatten(x)
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
            net = tf.contrib.layers.batch_norm(
                                                                net,
                                                                center = True,
                                                                scale = True)
            tf.summary.histogram('activation', net)
            return net
           

def resi_net(x, hid_dims=[500, 300], num_classes = 2):
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
            net = tf.contrib.layers.batch_norm(
                                                                net,
                                                                center = True,
                                                                scale = True)
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
                net = tf.contrib.layers.batch_norm(
                                                                net,
                                                                center = True,
                                                                scale = True)
        return outputs
        
def CNN(x, num_filters=[16, 32, 64], seq_len=10240, width=1, num_classes = 2):
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


def DeepConvLSTM(x, num_filters=[64, 64], filter_size=5, num_lstm=128, seq_len=1280, width=2, num_classes = 2):
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


def RNN(x, num_lstm=128, seq_len=1280, width=2, num_classes = 2):
    '''Use RNN
    x: shape[batch_size,time_steps,n_input]'''
    with tf.variable_scope("rnn_lstm") as layer_scope:
        net = tf.reshape(x, [-1, seq_len, width])
        #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
        net = tf.unstack(net, seq_len, 1)
        ##### defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm, forget_bias=1)
        outputs, _ =tf.nn.static_rnn(lstm_layer, net, dtype="float32")

        out_weights=tf.Variable(tf.random_normal([num_lstm,num_classes]))
        out_bias=tf.Variable(tf.random_normal([num_classes]))
        net=tf.matmul(outputs[-1], out_weights) + out_bias
        #net = tf.layers.dense(inputs=net[ -1], units=num_classes, activation=tf.nn.sOFTMAX)
        tf.summary.histogram('activation', net)
    return net

def Dilated_CNN(x, num_filters=16, dilation_rate=[2, 8, 16], kernel_size = [5, 1], pool_size=[2, 1], pool_strides=[2, 2], seq_len=10240, width=1, num_classes = 10):
    '''Perform convolution on 1d data
    Atrous Spatial Pyramid Pooling includes:
        (a) one 1*1 convolution and three 3*3 convolutions with rates = (2,8,16) when output stride =16,
        all with 256 filters and batch normalization,
        (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    x: shape [batch_size, height, width, channels]
    feature_map_size (4,)
    image_level_features (?, 1, 1, 1)
    image_level_features (?, 1, 1, 8)
    image_level_features (?, ?, ?, 8)
    at_pool1x1 (?, 28, 28, 8)
    at_pool3x3_1 (?, 28, 28, 8)
    at_pool3x3_2 (?, 28, 28, 8)
    concat net  (?, 28, 28, 32)
    net  (?, 28, 28, 2)
    net  (?, 1568)
    '''
    ## Input layer
    net = tf.reshape(x,  [-1, seq_len, width, 1])
    #net = inputs
    feature_map_size = tf.shape(net)
    print "feature_map_size", feature_map_size.shape
    # apply global average pooling
    image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
    print "image_level_features", image_level_features.shape
    image_level_features = tf.layers.conv2d(
                                                                        inputs = image_level_features,
                                                                        filters = num_filters,
                                                                        kernel_size=[1, 1],
                                                                        activation=tf.nn.relu)
    print "image_level_features", image_level_features.shape
    image_level_features = tf.image.resize_bilinear(
                                                                image_level_features,
                                                                (feature_map_size[1], feature_map_size[2])
                                                                )
    print "image_level_features", image_level_features.shape
    # dialted Convolutional Layer
    #### dilated layers
    at_pool1x1 = tf.layers.conv2d(
                                                 inputs = net,
                                                 filters = num_filters,   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size =kernel_size,
                                                 padding = 'same',
                                                 activation = tf.nn.relu)
    print "at_pool1x1", at_pool1x1.shape
    at_pool3x3_1 = tf.layers.conv2d(
                                                 inputs = net,
                                                 filters = num_filters,   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size = kernel_size,
                                                 dilation_rate = (2, 1),
                                                 padding = 'same',
                                                 activation = tf.nn.relu)
    print "at_pool3x3_1", at_pool3x3_1.shape
    at_pool3x3_2 = tf.layers.conv2d(
                                                 inputs = net,
                                                 filters = num_filters,   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size = kernel_size,
                                                 dilation_rate = (8, 1),
                                                 padding = 'same',
                                                 activation = tf.nn.relu)
    print "at_pool3x3_2", at_pool3x3_2.shape
    at_pool3x3_3 = tf.layers.conv2d(
                                                 inputs = net,
                                                 filters = num_filters,   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size = kernel_size,
                                                 dilation_rate = (16, 1),
                                                 padding = 'same',
                                                 activation = None)
    print "at_pool3x3_3", at_pool3x3_3.shape
    net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                                name="concat")
    print "concat net ", net.shape
    net = tf.layers.conv2d(
                                                 inputs = net,
                                                 filters = num_filters / 4,   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size = [1, 1],
                                                 padding = 'same',
                                                 activation = tf.nn.relu)
    print "net ", net.shape
    tf.summary.histogram('activation', net)
    net = tf.contrib.layers.batch_norm(net, center = True, scale = True)
    net = tf.layers.flatten(net )
    print "net ", net.shape
    net = tf.layers.dense(inputs = net, units=200, activation = tf.nn.relu)
    ### Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    '''
    batch 0 loss 2.3174462 accuracy 0.07
    batch 10 loss 1.7063526 accuracy 0.69
    batch 50 loss 1.5800242 accuracy 0.88
    batch 100 loss 1.5421643 accuracy 0.92
    batch 150 loss 1.5268363 accuracy 0.94
    batch 200 loss 1.5248604 accuracy 0.91'''
    return logits
