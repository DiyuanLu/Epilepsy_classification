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
    print("flatten net", net.shape)
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
        print("within block", net.shape)
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

    print("b4_blocks", net.shape)
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
                print('resi', jj,"net", net.shape)
                #if jj < 2:
                    #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
                    #print('resi', jj,"net", net.shape)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
            #### high-way net
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH{}".format(jj))
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
            C = 1. - T
            net = H * T + net * C
    print("net", net.shape)

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
    print( net.shape)
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
            print('conv{}'.format(layer_id), net.shape)
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
        print("net ", net.shape)
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        net = tf.unstack(net, axis=1)

        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm)
        outputs, _ = tf.nn.static_rnn(lstm_layer, net, dtype=tf.float32)  ###net 
    with tf.variable_scope("dense_out") as layer_scope:
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.softmax)
        print("net ", net.shape)
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net ", net.shape)
        tf.summary.histogram('activation', net)
        print('final output', net.shape)

    return net


def RNN(x, num_lstm=128, seq_len=1240, width=2, group_size=32, num_classes = 2):
    '''Use RNN
    x: shape[batch_size,time_steps,n_input]'''
    with tf.variable_scope("rnn_lstm") as layer_scope:
        net = tf.reshape(x, [-1, seq_len, width])
        #ipdb.set_trace()
        #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
        net = tf.reshape(net, [-1, seq_len//group_size,group_size*2])   ### feed not only one row but 8 rows of raw data
        print("net ", net.shape)
        net = tf.unstack(net, axis=1)

        ##### defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_lstm, forget_bias=1)
        outputs, _ =tf.nn.static_rnn(lstm_layer, net, dtype="float32")
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        print("net ", net.shape)
        net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.softmax)
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        net = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net", net.shape)
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
    
    def atrous_spatial_pyramid_pooling(net, scope, depth=256):
    """
    ASPP consists of (a) one 1*1 convolution and three 3*3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """

    with tf.variable_scope(scope):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keep_dims=True)
        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1*1", activation_fn=None)
        image_level_features = tf.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1*1 = slim.conv2d(net, depth, [1, 1], scope="conv_1*1_0", activation_fn=None)

        at_pool3*3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3*3_1", rate=6, activation_fn=None)

        at_pool3*3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3*3_2", rate=12, activation_fn=None)

        at_pool3*3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3*3_3", rate=18, activation_fn=None)

        net = tf.concat((image_level_features, at_pool1*1, at_pool3*3_1, at_pool3*3_2, at_pool3*3_3), a*is=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1*1_output", activation_fn=None)
        return net

    '''
    ## Input layer
    inputs = tf.reshape(x,  [-1, seq_len, width, 1])
    net = inputs
    feature_map_size = tf.shape(net)
    print("feature_map_size", feature_map_size.shape)
    # apply global average pooling
    image_level_features = tf.reduce_mean(net, [2, 1], name='image_level_global_pool', keep_dims=True)
    print("image_level_features", image_level_features.shape)
    image_level_features = tf.layers.conv2d(
                                            inputs = image_level_features,
                                            filters = num_filters_cnn,
                                            kernel_size=[1, 1],
                                            activation=tf.nn.relu)
    print("image_level_features", image_level_features.shape)
    #ipdb.set_trace()
    #net = tf.layers.max_pooling2d(inputs=image_level_features, pool_size=[2, 1], strides=[2, 1])
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    ###net = tf.image.resize_bilinear(
                                    ###image_level_features,
                                    ###(feature_map_size[1], feature_map_size[2])
                                    ###)
    ###print("image_level_pool", net.shape)
    
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
            print("net ", ind, net.shape)

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
            print("atrous net ", ind, net.shape)
            pyramid_pool_feature.append(net)
    #ipdb.set_trace()
    net = tf.concat(([pyramid_pool_feature[i] for i in range( len(pyramid_pool_feature))]), axis=3,
                                name="concat")
    print("pyrimid features", ind, net.shape)
    net = tf.layers.conv2d(
                             inputs = net,
                             filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             kernel_size = [1, 1],
                             padding = 'same',
                             activation = tf.nn.relu)
    print("net conv2d ", net.shape)
    #tf.summary.histogram('activation', net)
    #net = tf.layers.conv2d(
                             #inputs = net,
                             #filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             #kernel_size = [1, 1],
                             #padding = 'same',
                             #activation = None)
    #print("net conv2d ", net.shape)
    '''########### Dense layer ##################3'''
    net = tf.layers.flatten(net )
    #ipdb.set_trace()
    #net = tf.reshape(net, [-1, 1] )
    print("flatten net ", net.shape)
    net = tf.layers.dense(inputs = net, units=50, activation = tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    print("net ", net.shape)
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

    print("b4_blocks", net.shape)
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
                print('resi', jj,"net", net.shape)
                if jj < 2:
                    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
                    print('resi', jj,"net", net.shape)
        #### high-way net
        H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.relu, name="denseH{}".format(jj))
        T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
        C = 1. - T
        net = H * T + net * C
    print("net", net.shape)

    net = tf.layers.batch_normalization(net, center = True, scale = True)
    
    ### all the segments comes back together
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    
    print("net unite", net.shape)
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits

def PyramidPoolingConv(x, num_filters=[2, 4, 8, 16, 32, 64, 128], filter_size=5, dilation_rate=[2, 8, 16, 32, 64], seq_len=10240, width=2, num_seg=5, num_classes=2):
    '''extract temporal dependencies with different levels of dilation'''
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
        print("input net", net.shape)
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
            print("net", net.shape)
    
        ## last layer 1*1 conv
        net = tf.layers.conv2d(
                                         inputs = net,
                                         filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         padding = 'same',
                                         activation = tf.nn.relu)
        net = tf.layers.batch_normalization(net, center = True, scale = True)

        print("last net", net.shape)
        pyramid_feature.append(net)
    
    net = tf.concat(([pyramid_feature[i] for i in range( len(pyramid_feature))]), axis=3,name="concat")
    print("pyrimid features", net.shape)
    net = tf.layers.conv2d(
                             inputs = net,
                             filters = 1,   ##[filter_height, filter_width, in_channels, out_channels]
                             kernel_size = [1, 1],
                             padding = 'same',
                             activation = tf.nn.relu)
    print("last last net", net.shape)

    '''########### Dense layer ##################3'''
    #net = tf.layers.flatten(net, [-1, ] )
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    print("flatten net ", net.shape)

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
        print("net1x1 in reduce", net.shape)
        net = tf.layers.conv2d(
                            inputs = net,
                            filters = convbig_filters[ind], 
                            kernel_size = [num_filters[ind], num_filters[ind]],   ### seq: 1
                            padding = 'same',
                            activation = tf.nn.relu)
        print("net{}x1 in reduce".format(num_filters[ind]), net.shape)
        filter_concat.append(net)

    ## pooling + 1 conv
    net = tf.layers.max_pooling2d(
                        inputs = inputs, 
                        pool_size=[3, 3], 
                        padding = 'same',
                        strides=[1, 1])
    print("net reduce pooling", net.shape)
    net = tf.layers.conv2d(
                        inputs = net,
                        filters = 8, 
                        kernel_size = [1, 1],
                        padding = 'same',
                        activation = tf.nn.relu)
    filter_concat.append(net)
    inception = tf.nn.relu(tf.concat(([filter_concat[i] for i in range( len(filter_concat))]), axis=3,name="concat"))
    print("inception concat", inception.shape)
    
    net = tf.reshape(inception, [-1, inception.shape[1]*inception.shape[2]*inception.shape[3]])
    print("flatten net ", net.shape)
    net = tf.layers.dense(inputs=net, units=700, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    #net = tf.layers.dense(inputs=net, units=100, activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net)
    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    
    return logits