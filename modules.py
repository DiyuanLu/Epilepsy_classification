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
            net = tf.layers.dense(
                                    net,
                                    num_outputs,
                                    activation=tf.nn.relu,
                                    name=layer_scope.name+"_dense")
            tf.summary.histogram('fc_{}'.format(layer_id)+'_activation', net)
            net = tf.layers.batch_normalization(net, name=layer_scope.name+"_bn")
            tf.summary.histogram('fc_{}'.format(layer_id)+'BN_activation', net)
        with tf.variable_scope('fc_out') as scope:
            net = tf.layers.dense(
                                    net,
                                    num_classes,
                                    activation=tf.nn.sigmoid,
                                    name=scope.name)
            tf.summary.histogram('fc_out_activation', net)
            net = tf.layers.batch_normalization(net, name=scope.name+"_bn")
            tf.summary.histogram('fc_out_BN_activation', net)
            return net


def resi_net(x, hid_dims=[500, 300], num_classes = 2):
    '''tight structure of fully connected and residual connection
    x: [None, seq_len, width]'''
    net = tf.layers.flatten(x)
    print("flatten net", net.shape.as_list())
    for layer_id, num_outputs in enumerate(hid_dims):
        with tf.variable_scope('FcResi_{}'.format(layer_id)) as layer_scope:
            ### layer block #1 fully + high-way + batch_norm
            net = tf.layers.dense(
                                                                    net,
                                                                    num_outputs,
                                                                    activation=tf.nn.relu)
            net = tf.layers.batch_normalization(
                                                                net,
                                                                center = True,
                                                                scale = True)
            #### high-way net
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH1")
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT1")
            C = 1. - T
            net = H * T + net * C
            ### batch normalization
            net = tf.layers.batch_normalization(net)
            with tf.variable_scope('fc_out'):
                ### another fully conn
                outputs = tf.layers.dense(
                                            net,
                                            num_classes,
                                            activation=tf.nn.sigmoid)
                net = tf.layers.batch_normalization(net)
        return outputs


def resBlock_CNN(x, filter_size=9, num_filters=[4, 8, 16], stride=3, No=0):
    '''Construct residual blocks given the num of filter to use within the block
    reference:
    https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32'''
    net = x
    for num_outputs in num_filters:
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(
                                inputs = net,
                                filters = num_outputs,
                                kernel_size = [filter_size,1],
                                padding= "same",
                                activation = None)
        print("within block", net.shape.as_list())
    output = inputs + net
    return output
    
def Highway_Block_CNN(x, filter_size=9, num_filters=[4, 8, 16], No=0):
    net = x
    with tf.variable_scope("highway_block"+str(No)):
        H = tf.layers.conv2d(
                                inputs = net,
                                filters = num_filters,
                                kernel_size = [filter_size,1],
                                activation = None)
        T = tf.layers.conv2d(inputs = net,
                                filters = num_filters,
                                kernel_size = [filter_size,1], #We initialize with a negative bias to push the network to use the skip connection
                                biases_initializer=tf.constant_initializer(-1.0),
                                activation=tf.nn.sigmoid)
        output = H*T + input_layer*(1.0-T)
        return output


def CNN(x, num_filters=[8, 8, 8], num_block=3, filter_size=[7, 2], seq_len=10240, width=1, num_classes = 2):
    '''Perform convolution on 1d data
    Param:
        x: input data, 3D,array, shape=[batch_size, seq_len, width]
        num_filters: number of filterse in one serial-conv-ayer, i,e. one block)
        num_block: number of residual blocks
    return:
        predicted logits
        '''

    # ipdb.set_trace()
    inputs = tf.reshape(x, [-1, seq_len, width, 1])   ###
    net = inputs
    variables = {}

    print("b4_blocks", net.shape.as_list())
    '''Construct residual blocks'''
    for jj in range(num_block): ### 
        with tf.variable_scope('Resiblock_{}'.format(jj)) as layer_scope:
            ### construct residual block given the number of blocks
            for ind, num_outputs in enumerate(num_filters):
                net = tf.layers.conv2d(
                                inputs = net,
                                filters = num_outputs,
                                kernel_size = filter_size,   ### using a  wider kernel size helps
                                strides = (2, 1),
                                padding = 'same',
                                activation=None, 
                                name = layer_scope.name+"_conv{}".format(ind))
                print('resi', jj,"net", net.shape.as_list())
                #if jj < 2:
                    #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
                    #print('resi', jj,"net", net.shape.as_list())
                net = tf.layers.batch_normalization(net, center = True, scale = True)
            #### high-way net
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH{}".format(jj))
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
            C = 1. - T
            net = H * T + net * C
    print("net", net.shape.as_list())

    net = tf.layers.batch_normalization(net, center = True, scale = True)
    
    ##### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])   ### *(10240//seq_len)get short segments together
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits





def DeepConvLSTM(x, num_filters=[8, 16, 32, 64], filter_size=9, num_lstm=64, group_size=32, seq_len=10240, width=2, num_classes = 2):
    '''work is inspired by
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    in-shape: (BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), if no sliding, then it's the length of the sequence
    another from https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    '''

    net = tf.reshape(x,  [-1, seq_len,  width,1])
    print( net.shape.as_list())
    for layer_id, num_outputs in enumerate(num_filters):
        with tf.variable_scope("block_{}".format(layer_id)) as layer_scope:
            net = tf.layers.batch_normalization(net, center = True, scale = True)
            net = tf.layers.conv2d(
                                                inputs = net,
                                                filters = num_outputs,
                                                kernel_size = [filter_size, 1],
                                                strides = (2, 1),
                                                padding = 'same',
                                                activation=tf.nn.relu
                                                )
            print('conv{}'.format(layer_id), net.shape.as_list())
            #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
            net = tf.layers.batch_normalization(net, center = True, scale = True)
            #### high-way net
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH{}".format(jj))
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
            C = 1. - T
            net = H * T + net * C
    #ipdb.set_trace()
    with tf.variable_scope("reshape4rnn") as layer_scope:
        ### prepare input data for rnn requirements. current shape=[None, seq_len, num_filters]
        ### Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        #ipdb.set_trace()
        net = tf.reshape(net, [-1, seq_len//group_size, width*num_filters[-1]*group_size])   ## group these data points together 
        print("net ", net.shape.as_list())
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        net = tf.unstack(net, axis=1)

        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm)
        outputs, _ = tf.nn.static_rnn(lstm_layer, net, dtype=tf.float32)  ###net 
    with tf.variable_scope("dense_out") as layer_scope:
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.softmax)
        print("net ", net.shape.as_list())
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net ", net.shape.as_list())
        tf.summary.histogram('activation', net)
        print('final output', net.shape.as_list())

    return net


def RNN(x, num_lstm=128, seq_len=1240, width=2, group_size=32, num_classes = 2):
    '''Use RNN
    x: shape[batch_size,time_steps,n_input]'''
    with tf.variable_scope("rnn_lstm") as layer_scope:
        net = tf.reshape(x, [-1, seq_len, width])
        #ipdb.set_trace()
        #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
        net = tf.reshape(net, [-1, seq_len//group_size,group_size*2])   ### feed not only one row but 8 rows of raw data
        print("net ", net.shape.as_list())
        net = tf.unstack(net, axis=1)

        ##### defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm, forget_bias=1)
        outputs, _ =tf.nn.static_rnn(lstm_layer, net, dtype="float32")
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        print("net ", net.shape.as_list())
        net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.softmax)
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net", net.shape.as_list())
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        tf.summary.histogram('activation', net)
    return net



def Atrous_CNN(x, num_filters_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [10, 1], seq_len=10240, width=1, num_classes = 10):
    '''Perform convolution on 1d data
    Atrous Spatial Pyramid Pooling includes:
    https://sthalles.github.io/deep_segmentation_network/
        (a) one 1*1 convolution and three 3*3 convolutions with rates = (2,8,16) when output stride =16,
        all with 256 filters and batch normalization,
        (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param
        net: tensor of shape [batch, in_height, in_width, in_channels].
        :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    x: shape [batch_size, height, width, channels]
    https://arxiv.org/abs/1706.05587
    '''
    ## Input layer
    inputs = tf.reshape(x,  [-1, seq_len, width, 1])
    net = inputs
    feature_map_size = tf.shape(net)
    print("feature_map_size", feature_map_size.shape.as_list())
    # apply global average pooling
    image_level_features = tf.reduce_mean(net, [2, 1], name='image_level_global_pool', keep_dims=True)
    print("image_level_features", image_level_features.shape.as_list())
    image_level_features = tf.layers.conv2d(
                                            inputs = image_level_features,
                                            filters = num_filters_cnn,
                                            kernel_size=[1, 1],
                                            activation=tf.nn.relu)
    print("image_level_features", image_level_features.shape.as_list())
    #ipdb.set_trace()
    #net = tf.layers.max_pooling2d(inputs=image_level_features, pool_size=[2, 1], strides=[2, 1])
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    ###net = tf.image.resize_bilinear(
                                    ###image_level_features,
                                    ###(feature_map_size[1], feature_map_size[2])
                                    ###)
    ###print("image_level_pool", net.shape.as_list())
    
    '''###################### Classic Convolutional Layer #################'''
    conv_net = inputs
    for ind, num_filter in enumerate(num_filters_cnn):
        with tf.variable_scope("block_{}".format(ind)) as layer_scope:
            conv_net = tf.layers.conv2d(
                                                inputs = conv_net,
                                                filters = num_filters_cnn[ind],
                                                kernel_size=kernel_size,
                                                padding = 'same',
                                                activation=tf.nn.relu)
            conv_net = tf.layers.max_pooling2d(inputs=conv_net, pool_size=[2, 1], strides=[2, 1])
            conv_net = tf.layers.batch_normalization(conv_net, center = True, scale = True)
            print("net ", ind, net.shape.as_list())

    '''####################### Atrous_CNN #####################'''
    pyramid_pool_feature = []
    for jj, rate in enumerate(dilation_rate):
        with tf.variable_scope("atrous_block_{}".format(jj + ind)) as layer_scope:
            #net = tf.nn.atrous_conv2d(
                                                        #net,
                                                        #[10, 1, 1, 1],   ##[filter_height, filter_width, in_channels, out_channels]
                                                        #[rate, rate],
                                                        #padding='same')
            net = tf.layers.conv2d(
                                                 inputs = conv_net,
                                                 filters = num_filters_cnn[-1],   ##[filter_height, filter_width, in_channels, out_channels]
                                                 kernel_size = kernel_size,
                                                 dilation_rate = (dilation_rate[jj], 1),
                                                 padding = 'same',
                                                 activation = None)
            net = tf.layers.batch_normalization(net, center = True, scale = True)
            print("atrous net ", ind, net.shape.as_list())
            pyramid_pool_feature.append(net)
    #ipdb.set_trace()
    net = tf.concat(([pyramid_pool_feature[i] for i in range( len(pyramid_pool_feature))]), axis=3,
                                name="concat")
    print("pyrimid features", ind, net.shape.as_list())
    net = tf.layers.conv2d(
                             inputs = net,
                             filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             kernel_size = [1, 1],
                             padding = 'same',
                             activation = tf.nn.relu)
    print("net conv2d ", net.shape.as_list())
    #tf.summary.histogram('activation', net)
    #net = tf.layers.conv2d(
                             #inputs = net,
                             #filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             #kernel_size = [1, 1],
                             #padding = 'same',
                             #activation = None)
    #print("net conv2d ", net.shape.as_list())
    '''########### Dense layer ##################3'''
    net = tf.layers.flatten(net )
    #ipdb.set_trace()
    #net = tf.reshape(net, [-1, 1] )
    print("flatten net ", net.shape.as_list())
    net = tf.layers.dense(inputs = net, units=50, activation = tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    print("net ", net.shape.as_list())
    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    ''''''
    return logits


def CNN_new(x, num_filters=[8, 16, 32], num_block=3, filter_size=9, seq_len=10240, width=1, num_seg=10, num_classes = 2):
    '''Perform convolution on 1d data
    Param:
        x: input data, 3D,array, shape=[batch_size, seq_len, width]
        num_filters: number of filterse in one serial-conv-ayer, i,e. one block)
        num_block: number of residual blocks
        num_seg: num of segements that divide ori data into
    return:
        predicted logits
        '''
    ## Input layer
    # ipdb.set_trace()
    net = tf.reshape(x, [-1, seq_len, width, 1])   ###
                                
    variables = {}

    print("b4_blocks", net.shape.as_list())
    '''Construct residual blocks'''
    for jj in range(num_block): ### 
        with tf.variable_scope('Resiblock_{}'.format(jj)) as layer_scope:
            ### construct residual block given the number of blocks
            for ind, num_outputs in enumerate(num_filters):
                net = tf.layers.conv2d(
                                inputs = net,
                                filters = num_outputs,
                                kernel_size = [filter_size, 1],   ### using a  wider kernel size helps
                                #strides = (2, 1),
                                padding = 'same',
                                activation=None, 
                                name = layer_scope.name+"_conv{}".format(ind))
                print('resi', jj,"net", net.shape.as_list())
                if jj < 2:
                    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
                    print('resi', jj,"net", net.shape.as_list())
        #### high-way net
        H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH{}".format(jj))
        T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
        C = 1. - T
        net = H * T + net * C
    print("net", net.shape.as_list())

    net = tf.layers.batch_normalization(net, center = True, scale = True)
    
    ### all the segments comes back together
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    
    print("net unite", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits

def PyramidPoolingConv(x, num_filters=[2, 4, 8, 16, 32, 64, 128], filter_size=5, dilation_rate=[2, 8, 16, 32, 64], seq_len=10240, width=2, num_seg=5, num_classes=2):
    '''extract temporal dependencies with different levels of dilation
    https://sthalles.github.io/deep_segmentation_network/'''
    inputs = tf.reshape(x,  [-1, seq_len, width, 1])

    pyramid_feature = []

    for jj, rate in enumerate(dilation_rate):
        ### get the dilated input layer for each level
        net = tf.layers.conv2d(
                                         inputs = inputs,
                                         filters = num_filters[0],   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         dilation_rate = (rate, 1),
                                         padding = 'same',
                                         activation = None)
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        net = tf.nn.relu(net)
        print("input net", net.shape.as_list())
        for ind, num_filter in enumerate( num_filters):
            ### start conv with each level dilation 
            with tf.variable_scope("dilate{}_conv{}".format(jj, ind)) as layer_scope:

                net = tf.layers.conv2d(
                                         inputs = net,
                                         filters = num_filter,   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         strides = (2, 1),
                                         padding = 'same',
                                         activation = tf.nn.relu)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
                #### high-way net
                H = tf.layers.dense(net, units=num_filter, activation=tf.nn.relu, name="denseH{}".format(ind))
                T = tf.layers.dense(net, units=num_filter, activation=tf.nn.sigmoid, name="denseT{}".format(ind))
                C = 1. - T
                net = H * T + net * C
            print("net", net.shape.as_list())
    
        ## last layer 1*1 conv
        net = tf.layers.conv2d(
                                         inputs = net,
                                         filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         padding = 'same',
                                         activation = tf.nn.relu)
        net = tf.layers.batch_normalization(net, center = True, scale = True)

        print("last net", net.shape.as_list())
        pyramid_feature.append(net)
    
    net = tf.concat(([pyramid_feature[i] for i in range( len(pyramid_feature))]), axis=3,name="concat")
    print("pyrimid features", net.shape.as_list())
    net = tf.layers.conv2d(
                             inputs = net,
                             filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             kernel_size = [1, 1],
                             padding = 'same',
                             activation = tf.nn.relu)
    print("last last net", net.shape.as_list())

    '''########### Dense layer ##################3'''
    #net = tf.layers.flatten(net, [-1, ] )
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    print("flatten net ", net.shape.as_list())

    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    return logits
    
def Inception(x, num_filters=[16, 32, 64, 128], filter_size=[5, 9],num_block=2, seq_len=10240, width=2, num_seg=5, num_classes=2):
    '''https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
    '''
    inputs = tf.reshape(x,  [-1, seq_len, width, 1])
    filter_concat = []
    net_1x1 = tf.layers.conv2d(
                            inputs = inputs,
                            filters = 8, 
                            kernel_size = [1, 1],
                            padding = 'same',
                            activation = tf.nn.relu)
    filter_concat.append(net_1x1)
    ## 5*1 conv level
    conv1x1_filters = [8, 4]
    convbig_filters = [16, 8]
    for ind, num_output in enumerate(filter_size):
        net = tf.layers.conv2d(
                            inputs = inputs,
                            filters = conv1x1_filters[ind], 
                            kernel_size = [1, 1],
                            padding = 'same',
                            activation = tf.nn.relu)
        print("net1x1 in reduce", net.shape.as_list())
        net = tf.layers.conv2d(
                            inputs = net,
                            filters = convbig_filters[ind], 
                            kernel_size = [filter_size[ind], 1],   ### seq: 1
                            padding = 'same',
                            activation = tf.nn.relu)
        print("net{}x1 in reduce".format(filter_size[ind]), net.shape.as_list())
        filter_concat.append(net)

    ## pooling + 1 conv
    net = tf.layers.max_pooling2d(
                        inputs = inputs, 
                        pool_size=[5, 1], 
                        padding = 'same',
                        strides=[1, 1])
    print("net reduce pooling", net.shape.as_list())
    net = tf.layers.conv2d(
                        inputs = net,
                        filters = 8, 
                        kernel_size = [1, 1],
                        padding = 'same',
                        activation = tf.nn.relu)
    filter_concat.append(net)
    inception = tf.nn.relu(tf.concat(([filter_concat[i] for i in range( len(filter_concat))]), axis=3,name="concat"))
    print("inception concat", inception.shape.as_list())
    
    net = tf.reshape(inception, [-1, inception.shape[1]*inception.shape[2]*inception.shape[3]])
    print("flatten net ", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=700, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    #net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net)
    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    
    #flatten features for fully connected layer
    inception2_flat = tf.reshape(inception2,[-1,28*28*4*map2])

    #Fully connected layers
    if train:
        h_fc1 =tf.nn.dropout(tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1),dropout)
    else:
        h_fc1 = tf.nn.relu(tf.matmul(inception2_flat,W_fc1)+b_fc1)
    
    logits = tf.matmul(h_fc1,W_fc2)+b_fc2
    return logits
    
    
    return logits


    
def Inception_complex(x, num_filters=[16, 32, 64, 128], filter_size=[5, 9],num_block=2, seq_len=10240, width=2, num_classes=2):
    '''https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
    '''
    inputs = tf.reshape(x,  [-1, seq_len, width, 1])
    for block in range(num_block):
        with tf.variable_scope('Incepblock_{}'.format(block)) as layer_scope:
            filter_concat = []
            ### branch No.1, image level
            net_1x1 = tf.layers.conv2d(
                                    inputs = inputs,
                                    filters = 8, 
                                    kernel_size = [1, 1],
                                    padding = 'same',
                                    activation = tf.nn.relu)
            print("{}net1x1".format(block), net_1x1.shape.as_list())
            max_net_1x1 = tf.layers.max_pooling2d(inputs=net_1x1, pool_size=[2, 1], strides=[2, 1])
            filter_concat.append(max_net_1x1)
            ## branch No.2/3, 5*1, 3*1 conv level
            conv1x1_filters = np.array([8, 4]) * (block + 1)
            convbig_filters = np.array([16, 8]) * (block + 1)
            for ind, num_output in enumerate(filter_size):
                net = tf.layers.conv2d(
                                    inputs = inputs,
                                    filters = conv1x1_filters[ind], 
                                    kernel_size = [1, 1],
                                    padding = 'same',
                                    activation = tf.nn.relu)
                print("{}net1x1 in reduce".format(block), net.shape.as_list())
                net = tf.layers.conv2d(
                                    inputs = net,
                                    filters = convbig_filters[ind], 
                                    kernel_size = [filter_size[ind], 1],   ### seq: 1
                                    padding = 'same',
                                    activation = tf.nn.relu)
                print("{}net{}x1 in reduce".format(block, filter_size[ind]), net.shape.as_list())
                net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
                filter_concat.append(net)

            ## branch No.4, pooling + 1 conv
            net = tf.layers.max_pooling2d(
                                inputs = inputs, 
                                pool_size=[filter_size[0], 1], 
                                padding = 'same',
                                strides=[1, 1])
            print("{}net reduce pooling".format(block), net.shape.as_list())
            net = tf.layers.conv2d(
                                inputs = net,
                                filters = 8, 
                                kernel_size = [1, 1],
                                padding = 'same',
                                activation = tf.nn.relu)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
            print("{}net reduce pooling- pooliing".format(block), net.shape.as_list())
            filter_concat.append(net)

            ### concat all feature maps from 4 branches
            inception = tf.nn.relu(tf.concat(([filter_concat[i] for i in range( len(filter_concat))]), axis=3,name="concat"))
            inputs = tf.layers.conv2d(
                                        inputs = inception,
                                        filters = 1,
                                        kernel_size = [1, 1],
                                        padding = 'same',
                                        activation = tf.nn.relu)   ## as the next block inputs
            print("{}inception- inception".format(block), inception.shape.as_list())
            print("{}inception- inputs".format(block), inputs.shape.as_list())
    
    net = tf.reshape(inception, [-1, inception.shape[1]*inception.shape[2]*inception.shape[3]])
    print("flatten net ", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=700, activation=tf.nn.relu)
    net = tf.nn.dropout(net, 0.75)
    #net = tf.layers.batch_normalization(net)
    #net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net)
    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    
    
    return logits


def ResNet(x, num_layer_per_block=3, num_block=4, num_filters=[32, 64, 128], seq_len=10240, width=2, num_classes=2):
    '''https://medium.com/@pierre_guillou/understand-how-works-resnet-without-talking-about-residual-64698f157e0c
    34-layer-residual structure'''
    net = tf.reshape(x, [-1, seq_len, width, 1])
    net = tf.layers.conv2d( 
                            inputs = net,
                            filters = 32,
                            kernel_size = [9, 9],
                            padding = 'same',
                            activation = tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
    net = tf.layers.batch_normalization(net)
    for block in range(num_block):
        net = tf.layers.conv2d( 
                                inputs = net,
                                filters = num_filters[block],
                                kernel_size = [5, 1],
                                padding = 'same',
                                activation = tf.nn.relu)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
        net = tf.layers.batch_normalization(net)
        for layer in range(num_layer_per_block):
            with tf.variable_scope("block{}_layer{}".format(block, layer)) as layer_scope:
                ### sub block No.1
                net = tf.layers.conv2d( 
                                    inputs = net,
                                    filters = num_filters[block],
                                    kernel_size = [5, 1],
                                    padding = 'same',
                                    activation = tf.nn.relu)
                net = tf.layers.batch_normalization(net)
                net = tf.layers.conv2d( 
                                    inputs = net,
                                    filters = num_filters[block],
                                    kernel_size = [5, 1],
                                    padding = 'same',
                                    activation = tf.nn.relu)
                net = tf.layers.batch_normalization(net)
                #### high-way net
                H = tf.layers.dense(net, units=num_filters[block], activation=tf.nn.relu)
                T = tf.layers.dense(net, units=num_filters[block], activation=tf.nn.sigmoid)
                C = 1. - T
                net = H * T + net * C
                net = tf.layers.batch_normalization(net)
    ### average pool
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
    net = tf.reshape(net, [-1, net.shape[1]*net.shape[2]*net.shape[3]])
    ### dense layer
    net = tf.layers.dense(net, units=500, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    logits = tf.layers.dense(net, units=num_classes, activation=tf.nn.relu)
    
    return logits
   
    
        
        
        
def SpatiotemporalAttention():
    '''Paper: Diversity regulatrized spatiotemporal atention for video-based person re-identification
    
    '''

