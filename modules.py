import numpy as np
import tensorflow as tf
import ipdb



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


def resi_net(x, hid_dims=[500, 300], seq_len=10240, width=2, channels=1, num_blocks=2, num_classes = 2):
    '''tight structure of fully connected and residual connection
    x: [None, seq_len, width]'''

    net = tf.layers.flatten(x)
    print("flatten net", net.shape.as_list())
    for block in range(num_blocks):
        with tf.variable_scope('FcResiBlock_{}'.format(block)) as layer_scope:
            ### layer block #1 fully + high-way + batch_norm
            out1 = Highway_Block_FNN(net, hid_dims=500, name=layer_scope.name)

            net = Highway_Block_FNN(out1, hid_dims=200, name=layer_scope.name)

    ### another fully conn
    outputs = tf.layers.dense(
                                net,
                                num_classes,
                                activation=tf.nn.sigmoid)

    return outputs

def Highway_Block_FNN(x, hid_dims=100, name='highway'):
    '''https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32'''
    #net = tf.layers.flatten(x)
    transform_x = tf.layers.dense(x, units=hid_dims, activation=tf.nn.relu)
    #print(name + 'transform_x', transform_x.shape.as_list())
    
    H = tf.layers.dense(x, units=hid_dims, activation=tf.nn.relu)
    print(name + 'H', H.shape.as_list())
    T = tf.layers.dense(x, units=hid_dims, activation=tf.nn.sigmoid)
    print(name + 'T', T.shape.as_list())
    C = 1. - T
    #ipdb.set_trace()
    output = tf.add(tf.multiply(H, T), tf.multiply(transform_x, C))  # y = (H * T) + (x * C)
    #output = H * T + x * C  # y = (H * T) + (x * C)
    print(name + 'output', output.shape.as_list())
    return output


    
def Highway_Block_CNN(x, filter_size=[9, 1], output_channels=8, No_block=0):
    '''highway CNN block https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32
    param:
        x: batch*seq_len*width*channels
        filter_size: kernel size to use in CNN
        output_channels: output channels of CNN
        No_block: the name number'''
    assert len(x.shape) > 2 ('Should input image-like shape')  ## to do conv using batch_size * height * width * channel

    with tf.variable_scope("highway_block"+str(No)):

        H = tf.layers.conv2d(
                                inputs = x,
                                filters = output_channels,
                                kernel_size = filter_size,
                                padding = 'same',
                                activation = tf.nn.relu)
        T = tf.layers.conv2d(
                                inputs = x,
                                filters = output_channels,
                                kernel_size = filter_size, #We initialize with a negative bias to push the network to use the skip connection
                                padding = 'same',
                                biases_initializer=tf.constant_initializer(-1.0),
                                activation=tf.nn.sigmoid)
        #output = tf.add(tf.multiply(H, T), tf.multiply(x, 1 - T), name='y')
        output = H * T + x *(1.0 - T)
        return output

### should be good! same conv in the stack No bottleneck
def resBlock_CNN(x, filter_size=[9, 1], num_stacks=3, output_channels=16, stride=[1, 1], name='block', No_block=0):
    '''Construct residual blocks given the num of filter to use within the block
    param:
        filter_size
        output_channels
        stride:
    reference:
    https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32'''
    
    #net = tf.layers.batch_normalization(x)
    #net = tf.nn.relu(net)

    ### 1x1, channels you want
    for stack in range(num_stacks):
        net = tf.layers.conv2d(
                                inputs = x,
                                filters = output_channels,
                                kernel_size = filter_size,
                                padding='same',
                                activation = None)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        ### 3x3 transformation, channels you want
        net = tf.layers.conv2d(
                                inputs = net,
                                filters = output_channels,
                                kernel_size = filter_size,
                                padding='same',
                                activation = None)
        net = tf.layers.batch_normalization(net)
        x = x + net  ### add residual connection
        x = tf.nn.relu(x)   ### updata the inputs for next stack
    

    return x

        
### should be good!
def CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], strides=[2, 2], seq_len=10240, width=1, channels=1, num_classes = 2):
    '''Perform convolution on 1d data
    tutorial how VNN works
    http://cs231n.github.io/convolutional-networks/
    Param:
        x: input data, 3D,array, shape=[batch_size, seq_len, width]
        output_channels: number of filterse in one serial-conv-ayer, i,e. one block)
        num_block: number of residual blocks
    return:
        predicted logits
        '''

    # ipdb.set_trace()
    inputs = tf.reshape(x, [-1, seq_len, width, channels])   ###
    net = inputs
    variables = {}
    net = tf.layers.conv2d(
                            inputs = net,
                            filters = output_channels[0],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = None)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.nn.relu(net)
    print("b4_blocks", net.shape.as_list())
    '''Construct residual blocks'''
    for jj in range(num_block): ### 
        with tf.variable_scope('Resiblock_{}'.format(jj)) as layer_scope:
            ### construct residual block given the number of blocks
                
            print("block_{}_start shape {}".format(jj, net.shape.as_list()) )

            net = resBlock_CNN(net, filter_size=filter_size, output_channels=output_channels[jj], stride=[1, 1], name = layer_scope.name, No_block=jj)
            print("block_{}_Out shape {}".format(jj, net.shape.as_list()) )

            if jj < num_block - 1:  ## don't do subsampling at the last blcok'
                #### subsampling
                
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = output_channels[jj+1],
                                        kernel_size = filter_size,
                                        strides = strides,
                                        padding= "same",
                                        activation = None)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = output_channels[jj+1],
                                        kernel_size = filter_size,
                                        padding= "same",
                                        activation = None)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
                net = tf.nn.relu(net)
                print("block_{}_subsampling shape {}".format(jj, net.shape.as_list()) )


    
    ##### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])   ### *(10240//seq_len)get short segments together
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits


def Plain_CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], pool_size=[2,2], strides=[2, 2], seq_len=10240, width=1, channels=1, num_classes = 2):
    '''https://gist.github.com/giuseppebonaccorso/e77e505fc7b61983f7b42dc1250f31c8
    A plain CNN as in the tutorial'''
    net = tf.layers.conv2d(
                            inputs = x,
                            filters = output_channels[0],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = tf.nn.relu)
    #net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dropout(inputs=net, rate=0.25)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.layers.conv2d(
                            inputs = net,
                            filters = output_channels[1],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = tf.nn.relu)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides)
    print("shape {}".format(net.shape.as_list()) )
    #net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dropout(inputs=net, rate=0.25)
    net = tf.layers.conv2d(
                            inputs = net,
                            filters = output_channels[2],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = tf.nn.relu)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.layers.conv2d(
                            inputs = net,
                            filters = output_channels[2],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = tf.nn.relu)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides)
    #net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dropout(inputs=net, rate=0.25)
    print("shape {}".format(net.shape.as_list()) )
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])   ### *(10240//seq_len)get short segments together
    net = tf.layers.dense(inputs=net, units=1024, activation=tf.nn.relu)
    #net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dropout(inputs=net, rate=0.5)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    print("shape {}".format(logits.shape.as_list()) )
    return logits
    




def DeepConvLSTM(x, output_channels=[8, 16, 32, 64], filter_size=[9, 1], num_lstm=64, group_size=32, seq_len=10240, width=2, channels=1, num_classes = 2):
    '''work is inspired by
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    in-shape: (BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), if no sliding, then it's the length of the sequence
    another from https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    '''

    net = tf.reshape(x,  [-1, seq_len,  width, channels])
    print( net.shape.as_list())
    for layer_id, num_outputs in enumerate(output_channels):
        with tf.variable_scope("block_{}".format(layer_id)) as layer_scope:
            net = tf.layers.batch_normalization(net, center = True, scale = True)
            net = tf.layers.conv2d(
                                                inputs = net,
                                                filters = num_outputs,
                                                kernel_size = filter_size,
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
        ### prepare input data for rnn requirements. current shape=[None, seq_len, output_channels]
        ### Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        #ipdb.set_trace()
        net = tf.reshape(net, [-1, seq_len//group_size, width*output_channels[-1]*group_size])   ## group these data points together 
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



def Atrous_CNN(x, output_channels_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], kernel_size = [9, 1], seq_len=10240, width=1, num_classes = 10):
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
                                            filters = output_channels_cnn,
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
    for ind, num_filter in enumerate(output_channels_cnn):
        with tf.variable_scope("block_{}".format(ind)) as layer_scope:
            conv_net = tf.layers.conv2d(
                                                inputs = conv_net,
                                                filters = output_channels_cnn[ind],
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
                                 filters = output_channels_cnn[-1],   ##[filter_height, filter_width, in_channels, out_channels]
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


def CNN_new(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], pool_size=[2, 1], strides=[2, 1], seq_len=10240, width=1, num_seg=10, num_classes = 2):
    '''Perform convolution on 1d data
    Param:
        x: input data, 3D,array, shape=[batch_size, seq_len, width]
        output_channels: number of filterse in one serial-conv-ayer, i,e. one block)
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
            for ind, num_outputs in enumerate(output_channels):
                net = tf.layers.conv2d(
                                inputs = net,
                                filters = num_outputs,
                                kernel_size = filter_size,   ### using a  wider kernel size helps
                                #strides = (2, 1),
                                padding = 'same',
                                activation=None, 
                                name = layer_scope.name+"_conv{}".format(ind))
                print('resi', jj,"net", net.shape.as_list())
                if jj < 2:
                    net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, strides=strides)
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

def PyramidPoolingConv(x, output_channels=[2, 4, 8, 16, 32, 64, 128], filter_size=[5, 1], dilation_rate=[2, 8, 16, 32, 64], seq_len=10240, width=2, channels=1, num_seg=5, num_classes=2):
    '''extract temporal dependencies with different levels of dilation
    https://sthalles.github.io/deep_segmentation_network/'''
    inputs = tf.reshape(x,  [-1, seq_len, width, channels])

    pyramid_feature = []

    for jj, rate in enumerate(dilation_rate):
        ### get the dilated input layer for each level
        net = tf.layers.conv2d(
                                         inputs = inputs,
                                         filters = output_channels[0],   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         dilation_rate = (rate, 1),
                                         padding = 'same',
                                         activation = None)
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        net = tf.nn.relu(net)
        print("input net", net.shape.as_list())
        for ind, num_filter in enumerate( output_channels):
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

    
def Inception_block(x, in_channels = 32, out_channels=32, reduce_chanenls=16, filter_size=[[5, 1], [9, 1]], pool_size=[2, 1]):
    '''https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
    '''
    reduce_chanenls = reduce_chanenls   ##number of feature maps output by each 1x1 convolution that precedes a large convolution
    #inputs = tf.reshape(x,  [-1, seq_len, width, channels])
    filter_concat = []
    ### branch No. 1 image level
    net_1x1 = tf.layers.conv2d(
                            inputs = x,
                            filters = out_channels, 
                            kernel_size = [1, 1],
                            padding = 'same',
                            activation = tf.nn.relu)
    print("net1x1", net_1x1.shape.as_list())
    filter_concat.append(net_1x1)
    
    ## branch No.2/3, 5*1, 3*1 conv level
    for ind, kernel in enumerate(filter_size):
        net = tf.layers.conv2d(
                            inputs = x,
                            filters = reduce_chanenls, 
                            kernel_size = [1, 1],
                            padding = 'same',
                            activation = tf.nn.relu)
        print("net1x1 in reduce", net.shape.as_list())
        net = tf.layers.conv2d(
                            inputs = net,
                            filters = out_channels, 
                            kernel_size = kernel,   ### seq: 1
                            padding = 'same',
                            activation = tf.nn.relu)
        print("net{}x1 in reduce".format(filter_size[ind]), net.shape.as_list())
        filter_concat.append(net)

    ## branch No.4, pooling + 1 conv
    net = tf.layers.max_pooling2d(
                        inputs = x, 
                        pool_size=pool_size, 
                        padding = 'same',
                        strides=[1, 1])
    print("net reduce pooling", net.shape.as_list())
    net = tf.layers.conv2d(
                        inputs = net,
                        filters = out_channels, 
                        kernel_size = [1, 1],
                        padding = 'same',
                        activation = tf.nn.relu)
                        
    print("net reduce pooling- pooliing", net.shape.as_list())
    filter_concat.append(net)

    #### concat all feature maps from 4 branches
    inception = tf.nn.relu(tf.concat(([filter_concat[i] for i in range( len(filter_concat))]), axis=3,name="concat"))
    print("inception concat", inception.shape.as_list())

    return inception


    
def Inception_complex(x, output_channels=[16, 32], filter_size=[[5, 1], [9, 1]], reduce_chanenls=16, pool_size=[2, 1], strides=[2, 1], num_blocks=2, seq_len=10240, width=2, channels=1, num_classes=2):
    '''https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
    '''
    
    x = tf.reshape(x,  [-1, seq_len, width, channels])

    ## build inception blocks
    in_channels = 1
    for block in range(num_blocks):        
        net = Inception_block(x, in_channels = in_channels, out_channels=output_channels[block], reduce_chanenls=reduce_chanenls, filter_size=filter_size, pool_size=pool_size)
        in_channels = net.shape[2]
    
    net = tf.reshape(net, [-1, net.shape[1]*net.shape[2]*net.shape[3]])
    print("flatten net ", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=700, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)

    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    
    
    return logits


def ResNet(x, num_layer_per_block=3, num_block=4, filter_size=[[5, 1], [9, 1]], pool_size=[2, 1], strides=[2, 1], output_channels=[32, 64, 128], seq_len=10240, width=2, channels=1, num_classes=2):
    '''https://medium.com/@pierre_guillou/understand-how-works-resnet-without-talking-about-residual-64698f157e0c
    34-layer-residual structure'''
    net = tf.reshape(x, [-1, seq_len, width, channels])
    net = tf.layers.conv2d( 
                            inputs = net,
                            filters = 32,
                            kernel_size = filter_size[1],
                            padding = 'same',
                            activation = None)
    #net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    for block in range(num_block):
        net = tf.layers.conv2d( 
                                inputs = net,
                                filters = output_channels[block],
                                kernel_size = filter_size[0],
                                padding = 'same',
                                activation = None)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size, strides=strides)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        for layer in range(num_layer_per_block):
            with tf.variable_scope("block{}_layer{}".format(block, layer)) as layer_scope:
                ### sub block No.1
                net = tf.layers.conv2d( 
                                    inputs = net,
                                    filters = output_channels[block],
                                    kernel_size = filter_size[0],
                                    padding = 'same',
                                    activation = None)
                net = tf.layers.batch_normalization(net)
                net = tf.nn.relu(net)
                net = tf.layers.conv2d( 
                                    inputs = net,
                                    filters = output_channels[block],
                                    kernel_size = filter_size[0],
                                    padding = 'same',
                                    activation = None)
                net = tf.layers.batch_normalization(net)
                net = tf.nn.relu(net)
                #### high-way net
                H = tf.layers.dense(net, units=output_channels[block], activation=tf.nn.relu)
                T = tf.layers.dense(net, units=output_channels[block], activation=tf.nn.sigmoid)
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
   

def Bottleneck_stage(x, in_channel, out_channel, num_stack=3, filter_size=[3, 3], strides=[2, 2], cardinality=8, name=0):
    '''one Aggregate block'''
    with tf.variable_scope("Bottleneck_stage_{}".format(name)) as layer_scope:
        for stack in range(num_stack):
            #with tf.variable_scope("Block_{}_stack_{}".format(name, stack)):
            ### cardinality conv
            aggregate_net = 0  ## residial connection
            for c in range(cardinality):                        
                ## bottle neck
                net = tf.layers.conv2d(
                                        inputs = x,
                                        filters = 4,
                                        kernel_size = [1, 1],
                                        padding = 'same',
                                        activation = None
                                    )
                net = tf.layers.batch_normalization(net)
                net = tf.nn.relu(net)
                
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = 4,
                                        kernel_size = filter_size,
                                        padding = 'same',
                                        activation = None
                                        )
                net = tf.layers.batch_normalization(net)
                net = tf.nn.relu(net)
                
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = in_channel,
                                        kernel_size = [1, 1],
                                        padding = 'same',
                                        activation = None
                                        )
                aggregate_net += net
            
            aggregate_net = tf.layers.batch_normalization(aggregate_net)
            aggregate_net = tf.nn.relu(aggregate_net)
            inputs = aggregate_net + x ### residual add

        ## "Width is increased by 2 when the stage changes (downsampling), as in Sec. 3.1"
        net = tf.layers.conv2d(
                                    inputs = net,
                                    filters = in_channel * 2,
                                    kernel_size = [1, 1],
                                    strides = strides,
                                    padding = 'same',
                                    activation = None
                                    )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        print("resi{}_output {}".format(name, stack), inputs.shape.as_list())

        return net
            

def AggResNet(x, output_channels=[2, 4, 8], num_stacks=[3, 4, 3], cardinality=16, seq_len=10240, width=2, channels=1, filter_size=[[11, 1], [9, 1]], pool_size=[2, 1], strides=[2, 1], num_classes=2, ifaverage_pool=False):
    '''Paper: Aggregated Residual Transformations for Deep Neural Networks
    param:
        output_channels: the output channesl for each block, and each block have a number of cardinality paths in parallel
        num_subBlock: the number of stacked subbloks, should be in the same size as output_channels

    ('starting 3x3 conv', [None, 32, 32, 16])
    ('Stage 0 start', [None, 32, 32, 16])
    ('resi0_output 2', [None, 32, 32, 16])
    ('Stage 1 start', [None, 16, 16, 32])
    ('resi1_output 3', [None, 16, 16, 32])
    ('Stage 2 start', [None, 8, 8, 64])
    ('resi2_output 2', [None, 8, 8, 64])

    '''
    net = tf.reshape(x, [-1, seq_len, width, channels])
    ### Conv 1 kernel_size=big, stride=2
    net = tf.layers.conv2d(
                            inputs = net,
                            filters = output_channels[0],
                            kernel_size = filter_size[0],
                            padding = 'same',
                            activation = None
                            )
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    print("starting 3x3 conv", net.shape.as_list())

        
    for ind, out_channel in enumerate(output_channels):
        print("Stage {} start".format(ind), net.shape.as_list())
        
        net = Bottleneck_stage(net, net.shape[-1], out_channel, num_stack=num_stacks[ind], cardinality=cardinality, name=ind)
    #ipdb.set_trace()    
    net = tf.layers.average_pooling2d(inputs=net, pool_size=pool_size, strides=[1, 1], padding='same')
    print("pooling", net.shape.as_list())
    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=500, activation=tf.nn.relu)
    net = tf.layers.batch_normalization(net)
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)

    return logits
        
        

def AggregatedResnet(x, output_channels=[32, 16, 8], num_stacks=[3, 4, 6, 3], cardinality=16, seq_len=10240, width=2, channels=1, filter_size=[[11, 1], [9, 1]], pool_size=[2, 1], strides=[2, 1], num_classes=2):
    '''https://blog.waya.ai/deep-residual-learning-9610bb62c355'''

    def add_common_layers(y):
        y = tf.layers.batch_normalization(y)
        y = tf.nn.leaky_relu(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return tf.layers.conv2d(inputs=y, filters=nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality    ### divide into groups and do conv

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(tf.layers.conv2d(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = tf.layers.conv2d(y, nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = tf.layers.conv2d(y, nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = tf.layers.batch_normalization(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1x1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = tf.layers.conv2d(shortcut, nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')
            shortcut = tf.layers.batch_normalization(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = tf.nn.leaky_relu(y)

        return y

    # conv1
    x = tf.layers.conv2d(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1)(x)

    return x
        
def SpatiotemporalAttention():
    '''Paper: Diversity regulatrized spatiotemporal atention for video-based person re-identification
    
    '''
    ### CNN extract features from data

    ### define K attention modules
    ##e(n,k,l) = [w'(s,k)]T*Relu(W(s,k)f(n,l) + b(s,k)) + b'(s,k),
    ## attention = softmax(e(n,k))

    ### 


































        ### pickle training data ###
        # ipdb.set_trace()
        # try:
        #     data_train_all = pickle.load(open('data/{}_pickle_data_train.p'.format(data_version), 'rb'))
        #     labels_train_all = pickle.load(open('data/{}_pickle_labels_train.p'.format(data_version), 'rb'))
        #     data_test_all = pickle.load(open('data/{}_pickle_data_test.p'.format(data_version), 'rb'))
        #     labels_test_all = pickle.load(open('data/{}_pickle_labels_test.p'.format(data_version), 'rb'))
        #     print("Loaded pickles!")
        # except:
        #
        #     data_train_all = np.zeros((num_train, 10240, 2))
        #     #labels_train_all = np.zeros((num_train))
        #     data_test_all = np.zeros((num_test, 10240, 2))
        #     #labels_test_all = np.zeros((num_test))
        #     for ind, filename in enumerate(files_train):
        #         data = func.read_data(filename, header=header, ifnorm=ifnorm)
        #         data_train_all[ind, :, :] = data
        #         #if 'F' in filename:
        #             #labels_train_all[ind] = 1
        #         if ind % 1000 == 0:
        #             print(ind, filename, labels_train[ind])
        #
        #     for ind, filename in enumerate(files_test):
        #         data = func.read_data(filename, header=header, ifnorm=ifnorm)
        #         data_test_all[ind, :, :] = data
        #         #if 'F' in filename:
        #             #labels_train_all[ind] = 1
        #         if ind % 100 == 0:
        #             print(ind, filename, labels_test[ind])
        #
        #     pickle.dump(data_train_all, open( 'data/{}_pickle_data_train.p'.format(data_version), 'wb'))
        #     pickle.dump(labels_train, open( 'data/{}_pickle_labels_train.p'.format(data_version), 'wb'))
        #     pickle.dump(data_test_all, open( 'data/{}_pickle_data_test.p'.format(data_version), 'wb'))
        #     pickle.dump(labels_test, open( 'data/{}_pickle_labels_test.p'.format(data_version), 'wb'))
        # ipdb.set_trace()
