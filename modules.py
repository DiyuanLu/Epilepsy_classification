import numpy as np
import tensorflow as tf
import ipdb
import os
import functions as func

regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
initializer = tf.contrib.layers.xavier_initializer()

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    https://www.tensorflow.org/guide/summaries_and_tensorboard"""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


def get_variables_from_graph(layer, *args):
    '''get the variables from the graph for further visualization
    param:
        layer: the output of the layer
        args: keyword arguments
                variable_names: '/kernel:0', '/weights:0', '/bias:0'

    e.g.  all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_scope.name)
        '''
    for arg in args:
        variable_name = os.path.split(layer.name)[0] + '/'+arg+':0'
        variable = tf.get_default_graph().get_tensor_by_name(variable_name)
        if len(variable.shape.as_list()) == 4:            
            # to tf.image_summary format [batch_size, height, width, channels]
            variable = tf.transpose(variable, [3, 0, 1, 2])
        
            tf.summary.image(variable_name, variable)
        tf.summary

    return variable

def put_kernels_on_grid (kernel, pad = 1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
    Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    https://gist.github.com/kukuruza/03731dc494603ceab0c5
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(np.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    print('Who would enter a prime number of filters')
                return (i, int(n / i))
                
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)

    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)  ### normalize the kernel

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')
    print("tf.pad x.shape", x.shape.as_list()) ##(7, 3, 8, 16)
    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad
    #ipdb.set_trace()
    channels = kernel.get_shape()[2]    ## in channels
    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2)) ###(16, 7, 3, 8)
    print("tf.transpose(x, (3, 0, 1, 2)) x.shape", x.shape.as_list())
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))   ###(4, 28, 3, 8)
    print("tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels])) x.shape", x.shape.as_list())
    # switch X and Y axes
    x = np.transpose(x, (0, 2, 1, 3))       ##(4, 3, 28, 8)
    print("tf.transpose(x, (0, 2, 1, 3)) x.shape", x.shape.as_list())
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))  ###(1, 12, 28, 8)
    print("tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels])) x.shape", x.shape.as_list())

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))   ###(28, 12, 8, 1)
    print("tf.transpose(x, (2, 1, 3, 0)) x.shape", x.shape.as_list())

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))   ###(1, 28, 12, 8)
    print("tf.transpose(x, (3, 0, 1, 2)) x.shape", x.shape.as_list())

    # scaling to [0, 255] is not necessary for tensorboard
    return x

def add_kernel_to_image_summary_with_scope(scope):
    '''given a scope, add all the kernels to image summary as picture in grid'''
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope.name)
    for ind, var in enumerate(trainables):
        if 'kernel' in var.name and 'conv' in var.name:       
            grid = put_kernels_on_grid(var, pad = 1)
            tf.summary.image(var.name, grid, max_outputs=grid.shape[-1])

def add_kernel_to_image_summary_from_kernels(kernels):
    '''given a collection dict of all trainable kernels,
    add all the kernels to image summary as picture in grid'''

    for ind, kernel in enumerate(kernels):
        if len(kernel.shape) == 4:
            grid = put_kernels_on_grid (kernel, pad = 1)
            tf.summary.image(var.name, grid, max_outputs=1)
        elif len(kernel.shape) == 2:            
            tf.summary.image(var.name, kernel, max_outputs=1)
            
#####################################################################################################3
def fc_net(x, hid_dims=[500, 300, 100], num_classes = 2):
    net = tf.layers.flatten(x)
    # Convolutional Layer
    for layer_id, num_outputs in enumerate(hid_dims):   ## avoid the code repetation
        with tf.variable_scope('fc_{}'.format(layer_id)) as layer_scope:
            net = tf.layers.dense(
                                    net,
                                    num_outputs,
                                    activation=tf.nn.leaky_relu,
                                    kernel_regularizer=regularizer,
                                    name=layer_scope.name+"_dense")
            tf.summary.histogram('fc_{}'.format(layer_id)+'_activation', net)
            net = tf.layers.batch_normalization(net, name=layer_scope.name+"_bn")
            tf.summary.histogram('fc_{}'.format(layer_id)+'BN_activation', net)
            net = tf.layers.dropout(inputs=net, rate=0.5)
        with tf.variable_scope('fc_out') as scope:
            net = tf.layers.dense(
                                    net,
                                    num_classes,
                                    activation=tf.nn.sigmoid,
                                    kernel_regularizer=regularizer,
                                    name=scope.name)
            tf.summary.histogram('fc_out_activation', net)
            net = tf.layers.batch_normalization(net, name=scope.name+"_bn")
            tf.summary.histogram('fc_out_BN_activation', net)
            net = tf.layers.dropout(inputs=net, rate=0.5)
            return net


def resi_net(x, hid_dims=[500, 300], seq_len=10240, width=2, channels=1, num_blocks=2, num_classes = 2):
    '''tight structure of fully connected and residual connection
    x: [None, seq_len, width]'''

    net = tf.layers.flatten(x)
    #net = tf.reshape(x, [-1, seq_len*2])
    print("flatten net", net.shape.as_list())
    for block in range(num_blocks):
        with tf.variable_scope('FcResiBlock_{}'.format(block)) as layer_scope:
            ### layer block #1 fully + high-way + batch_norm
            out1 = Highway_Block_FNN(net, hid_dims=hid_dims[0], name=layer_scope.name)

            net = Highway_Block_FNN(out1, hid_dims=hid_dims[1], name=layer_scope.name)

    ### another fully conn

    outputs_pre = tf.layers.dense(
                                net,
                                num_classes,
                                kernel_regularizer=regularizer,
                                activation=None)
    print("outputs_pre shape", outputs_pre.shape.as_list())
    outputs = tf.nn.softmax(outputs_pre)

    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


def Highway_Block_FNN(x, hid_dims=100, name='highway'):
    '''https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32'''
    #net = tf.layers.flatten(x)
    transform_x = tf.layers.dense(x, units=hid_dims, activation=tf.nn.leaky_relu)
    #print(name + 'transform_x', transform_x.shape.as_list())
    transform_x = tf.layers.dropout(inputs=transform_x, rate=0.5)
    
    H = tf.layers.dense(x, units=hid_dims, activation=tf.nn.leaky_relu)

    T = tf.layers.dense(x, units=hid_dims, activation=tf.nn.sigmoid)

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
                                activation = tf.nn.leaky_relu)
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
def resBlock_CNN(x, filter_size=[9, 1], num_stacks=3, output_channels=16, stride=[1, 1], name='block0'):
    '''Construct residual blocks given the num of filter to use within the block
    param:
        filter_size
        output_channels
        stride:
    reference:
    https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32'''
    
    #net = tf.layers.batch_normalization(x)
    #net = tf.nn.leaky_relu(net)

    ### 1x1, channels you want
    
    for stack in range(num_stacks):
        with tf.variable_scope(name+"_stack{}".format(stack)) as scope:
            net = tf.layers.conv2d(
                                    inputs = x,
                                    filters = output_channels,
                                    kernel_size = filter_size,
                                    padding='same',
                                    activation = None)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.leaky_relu(net)
            ### 3x3 transformation, channels you want
            net = tf.layers.conv2d(
                                    inputs = net,
                                    filters = output_channels,
                                    kernel_size = filter_size,
                                    padding='same',
                                    activation = None)
            net = tf.layers.batch_normalization(net)
            #x = x + net  ### add residual connection
            net = tf.concat([x, net], axis=3 )
            x = tf.nn.leaky_relu(x)   ### updata the inputs for next stack
            print(scope.name + "-out", x.shape.as_list())

    return x

        
### should be good!
def CNN(x, output_channels=[8, 16, 32], num_block=3, filter_size=[9, 1], strides=[2, 2], fc=[300], seq_len=10240, width=1, channels=1, num_classes = 2):
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
    x = tf.reshape(x, [-1, seq_len, width, channels])   ###
    #net = inputs
    variables = {}
    net = tf.layers.conv2d(
                            inputs = x,
                            filters = output_channels[0],
                            kernel_size = filter_size,
                            padding= "same",
                            activation = tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    #net = tf.nn.leaky_relu(net)
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
                net = tf.nn.leaky_relu(net)
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = output_channels[jj+1],
                                        kernel_size = filter_size,
                                        padding= "same",
                                        activation = None)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
                net = tf.nn.leaky_relu(net)
                print("block_{}_subsampling shape {}".format(jj, net.shape.as_list()) )


    
    ##### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]])   ### *(10240//seq_len)get short segments together
    for ind, units in enumerate(fc):
        net = tf.layers.dense(inputs=net, units=units, kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)            
        net = tf.layers.dropout(net, rate=0.5)###0.50
    tf.summary.histogram("pre_activation", net)
    
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


def CNN_old(x, num_filters=[8, 16, 32], num_block=3, filter_size=9, seq_len=10240, width=1, fc=[300], num_classes = 2):
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
                                kernel_size = [filter_size, 1],   ### using a  wider kernel size helps
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
            H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.leaky_relu, name="denseH{}".format(jj))
            T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
            C = 1. - T
            net = H * T + net * C
    print("net", net.shape)

    net = tf.layers.batch_normalization(net, center = True, scale = True)
    
    ##### Logits layer
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*(10240//seq_len)])   ### get short segments together
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.leaky_relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


def DilatedCNN_Tutorial(x, output_channels=[32, 64, 128], seq_len=32, width=32, channels=3, pool_size=[2, 2], strides=[2, 2], filter_size=[3, 3], num_classes=10):
    '''https://github.com/exelban/tensorflow-cifar-10/blob/master/include/model.py'''
    x = tf.reshape(x, [-1, seq_len, width, channels])   ###
    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
            inputs=x,
            filters=output_channels[0],
            kernel_size=filter_size,
            strides=strides,
            padding='SAME',
            activation=tf.nn.leaky_relu
        )
        print("shape", conv.shape.as_list())
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=output_channels[1],
            dilation_rate = 2,
            kernel_size=filter_size,
            padding='SAME',
            activation=tf.nn.leaky_relu
        )
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=output_channels[1],
            dilation_rate = 4,
            kernel_size=filter_size,
            padding='SAME',
            activation=tf.nn.leaky_relu
        )
        print("shape", conv.shape.as_list())
        pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)
        print("shape", drop.shape.as_list())
        
    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=output_channels[2],
            kernel_size=filter_size,
            padding='SAME',
            activation=tf.nn.leaky_relu
        )
        print("shape", conv.shape.as_list())
        pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        print("shape", pool.shape.as_list())
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=output_channels[2],
            kernel_size=filter_size,
            strides=strides,
            padding='SAME',
            activation=tf.nn.leaky_relu
        )
        print("shape", conv.shape.as_list())
        pool = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        print("shape", pool.shape.as_list())
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)

    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, drop.shape[1]*drop.shape[2]*drop.shape[3]])
        print("shape", flat.shape.as_list())
        fc = tf.layers.dense(inputs=flat, units=1000, activation=tf.nn.leaky_relu)
        drop = tf.layers.dropout(fc, rate=0.5)
        logits = tf.layers.dense(inputs=drop, units=num_classes, activation=tf.nn.softmax, name=scope.name)



    return logits



def CNN_Tutorial(x, output_channels=[32, 64, 128], seq_len=32, width=32, channels=3, pool_size=[2, 2], strides=[2, 2], filter_size=[3, 3], num_classes=10, fc=[500], iffusion=True, num_seg=1):
    '''https://github.com/exelban/tensorflow-cifar-10/blob/master/include/model.py'''
    x = tf.reshape(x, [-1, seq_len, width, channels])   ###
    #for ind in range(3):
        #tf.summary.image('image_ori{}'.format(ind), tf.reshape(x[ind,...], [-1, seq_len, width, channels]))
        
    activities = {}
    with tf.variable_scope('conv1') as scope:
        net = tf.layers.conv2d(
            inputs=x,
            filters=output_channels[0],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=None
        )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #ipdb.set_trace()
        #add_kernel_to_image_summary_with_scope(scope)
        #func.add_conved_image_to_summary(conv)
        print(scope.name + "shape", net.shape.as_list())
    with tf.variable_scope('conv2_pool') as scope:
        net = tf.layers.conv2d(
            inputs=net,
            filters=output_channels[1],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=None
        )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.leaky_relu(net)
        #activities[net.name] = net
        #func.add_conved_image_to_summary(conv)
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        net = tf.layers.dropout(net, rate=0.25, name=scope.name)###0.25
        print(scope.name + "shape", net.shape.as_list())
        
    with tf.variable_scope('conv3') as scope:
        net = tf.layers.conv2d(
            inputs=net,
            filters=output_channels[2],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=None
        )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #func.add_conved_image_to_summary(conv)
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        
    with tf.variable_scope('conv3_pool') as scope:
        net = tf.layers.conv2d(
            inputs=net,
            filters=output_channels[2],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=None
        )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #func.add_conved_image_to_summary(conv)
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        net = tf.layers.dropout(net, rate=0.25, name=scope.name)###0.25
        print(scope.name + "shape", net.shape.as_list())
        
    with tf.variable_scope('conv4_pool') as scope:
        net = tf.layers.conv2d(
            inputs=net,
            filters=output_channels[2],
            kernel_size=filter_size[1],    #[2, 2],  #
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=None
        )
        net = tf.layers.batch_normalization(net)
        net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        print(scope.name + "shape", net.shape.as_list())
        net = tf.layers.dropout(net, rate=0.25, name=scope.name)   ###0.25

    with tf.variable_scope('fully_connected') as scope:
        net = tf.reshape(net, [-1, net.shape[1]*net.shape[2]*net.shape[3]])
        print(scope.name + "shape", net.shape.as_list())
        
        for ind, units in enumerate(fc):
            net = tf.layers.dense(inputs=net, units=units, kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
            activities[net.name] = net
            net = tf.layers.dropout(net, rate=0.5)
            
            print(scope.name + "shape", net.shape.as_list())
        #tf.summary.histogram("dense_out", net)
        #ipdb.set_trace()
        variable_summaries(tf.trainable_variables()[-2])
        
    #ipdb.set_trace()
    kernels = {}   #### implement attention 
    #if iffusion:
        #with tf.variable_scope('fusion') as scope:
            #fusion_w = tf.Variable(tf.random_normal([num_seg, num_classes], stddev=0.1), name="fusion_w")
            #fusion_b = tf.Variable(tf.random_normal([num_seg], stddev=0.1), name="fusion_b")
            #fusion_logits = tf.reshape(logits, [-1, num_seg, num_classes])
            #logits = tf.nn.softmax(tf.multiply(fusion_logits, fusion_w) + fusion_b)
            #kernels['fusion_w'] = fusion_w
            
    logits = tf.layers.dense(
                            inputs=net,
                            units=num_classes,
                            activation=tf.nn.softmax,
                            kernel_regularizer=regularizer,
                            name=scope.name)
    
    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    for var in all_trainable_vars:
        if 'kernel' in var.name:            
                kernels[var.name] = var
            
       
    return logits, kernels, activities


def CNN_Tutorial_attention(x, output_channels=[8, 16, 32], seq_len=10240, width=2, channels=1, pool_size=[4, 1], strides=[4, 1], filter_size=[5, 1], num_att=3, num_classes=2, fc=[150], att_dim=128, gird_height=5, gird_width=1, ifattnorm=True):
    ''' Network with attention is used together with segmenting original data into several segments.
    https://github.com/exelban/tensorflow-cifar-10/blob/master/include/model.py
    with attention module adapted from http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Diversity_Regularized_Spatiotemporal_CVPR_2018_paper.pdf
    Segment the input sequence into num_seg segmentations
    Initialize num_att attention modules in each segmentation, compute the attention weighted features of each attention module. COncat the gated feature from all attentino modules to form a super feature vector for FC
    Param:
        num_att: the number of attention modules. 'K'
        num_seg: the number of sub_segments of one input sample, each segment has num_att attention modules. 'L': temporal 'grid' cell
        att_dim: the lower attention dimention 'd'
    '''
    x = tf.reshape(x, [-1, seq_len, width, channels])   ###
    #for ind in range(3):
        #tf.summary.image('image_ori{}'.format(ind), tf.reshape(x[ind,...], [-1, seq_len, width, channels]))
        
    activities = {}
    with tf.variable_scope('conv1') as scope:
        net = tf.layers.conv2d(
            inputs=x,
            filters=output_channels[0],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        net = tf.layers.batch_normalization(net)
        #net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #ipdb.set_trace()
        #add_kernel_to_image_summary_with_scope(scope)
        #func.add_conved_image_to_summary(conv)
        print(scope.name + "shape", net.shape.as_list())
    with tf.variable_scope('conv2_pool') as scope:
        net = tf.layers.conv2d(
            inputs=net,
            filters=output_channels[1],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        net = tf.layers.batch_normalization(net)
        #net = tf.nn.leaky_relu(net)
        #activities[net.name] = net
        #func.add_conved_image_to_summary(conv)
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        pool = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)###0.25
        print(scope.name + "shape", drop.shape.as_list())

    with tf.variable_scope('conv3_pool') as scope:
        net = tf.layers.conv2d(
            inputs=drop,
            filters=output_channels[2],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        net = tf.layers.batch_normalization(net)
        #net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #func.add_conved_image_to_summary(conv)
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        pool = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)###0.25
        print(scope.name + "shape", pool.shape.as_list())
        
    with tf.variable_scope('conv4_pool') as scope:
        net = tf.layers.conv2d(
            inputs=pool,
            filters=output_channels[2],
            kernel_size=[2, 2],  #filter_size[1],    #
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        net = tf.layers.batch_normalization(net)
        #net = tf.nn.leaky_relu(net)
        #func.add_conved_image_to_summary(net)
        #activities[net.name] = net
        #add_kernel_to_image_summary_with_scope(scope)
        print(scope.name + "shape", net.shape.as_list())
        pool = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
        print(scope.name + "shape", pool.shape.as_list())
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)   ###0.25

    
    kernels = {}   #### implement attention 
    ## attention
    ## attention = att_w2 * tf.nn.relu(att_w1 * features_of_segment + att_b1) + att_b2
    ## D: the shape of flattened feature map, d: target dimension of the attention
    ## grid: here 5 x 4, Height = 5, width = 1, original 8 x 4
    K = num_att
    with tf.variable_scope('attention') as scope:
        net = tf.reshape(drop, [-1, gird_height, gird_width, drop.shape[1]*drop.shape[2]*drop.shape[3] // gird_height*gird_width])  ## [None, 5, 1, 1971]
        print(scope.name + "shape", net.shape.as_list())  ##('attentionshape', [None, L=5, 1, D=2048])
        D = net.shape.as_list()[3]
        
        e_att_1 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', activation=None, name='attConv_1')  ##(-1, L, 1, 64)
        e_att_1 = tf.layers.batch_normalization(e_att_1)
        e_att_1 = tf.nn.relu(e_att_1)#(-1, L, 1, 64)
        print("e_att 1 shape: ", e_att_1.shape.as_list())
        e_att_1 = tf.layers.conv2d(inputs=e_att_1, filters=K, kernel_size=[1, 1], strides=[1, 1], padding='valid', activation=None, name='attConv_2')  ##(-1, L, 1, K)
        e_att_1 = tf.layers.batch_normalization(e_att_1)
        e_att_1 = tf.nn.relu(e_att_1)#(-1, L, 1, K)
        print("e_att 2 shape: ", e_att_1.shape.as_list())
        e_att_1 = tf.reshape(e_att_1, [-1, num_att, gird_height*gird_width])  # (-1, K, L)
        ## get attention
        s_att = tf.nn.softmax(e_att_1, name='s_att')  ## (-1, K, L)
        print("s_att shape: ", s_att.shape.as_list())
        activities[s_att.name] = s_att
        
        # diversity regularization
        diversity = s_att
        tf.summary.histogram('attention', s_att)
        
        ## Multiple spatial attention
        s_att = tf.reshape(s_att, [-1, num_att, 1, gird_height, gird_width])  #(-1, K, 1, h, w) hxw=L
        s_att_repeat = tf.tile(s_att, [1, 1, D, 1, 1])   ##(-1, K, D, h, w)

        print("s_att shape: ", s_att_repeat.shape.as_list())
        net = tf.reshape(net, [-1, 1, D, gird_height, gird_width])   ##shape [-1, 1, D, L]
        ## Repeat net to get shape [-1, K, D, h, w]
        net_repeat = tf.tile(net, [1, K, 1, 1, 1])        ##[-1, K, D, h, w]
        print("net_repeat shape: ", net_repeat.shape.as_list())
        ### Attention gated feature
        net = net_repeat * s_att_repeat      #[-1, K, D, h, w]
        print("attention weighted features: ", net.shape.as_list())
        net = tf.reshape(net, [-1, D, gird_height, gird_width]) ##[-1*K, D, h, w]
        #net = tf.transpose(net, [0, 2, 3, 1])
        print("before average: ", net.shape.as_list())
        net = tf.transpose(net, [0, 2, 3, 1])   #[-1*K, h, w, D]
        #ipdb.set_trace()
        net = tf.layers.average_pooling2d(net, [net.shape.as_list()[1], net.shape.as_list()[2]], strides=1) * net.shape.as_list()[1] * net.shape.as_list()[2] ##[-1*K, 1, 1, D]
        net = tf.transpose(net, [0, 3, 1, 2])   #[-1*K, D, 1, 1]
        print("averaged: ", net.shape.as_list())
        net = tf.reshape(net, [-1, D])  ##[-1*K, D]

        net = tf.layers.dense(inputs=net, units=att_dim, activation=None)   #[-1*K, att_dim]
        net = tf.reshape(net, [-1, K, att_dim])##[-1, K, att_dim]
               
        if ifattnorm:
            net = tf.norm(net, ord=2, axis=2, keep_dims=True)  ##[-1, K, 1]
            net = tf.tile(net, [1, 1, att_dim])  #[-1, K, att_dim]
            
    with tf.variable_scope('fc_out') as scope:
        ### after attention gating, pass to FC layer
        net = tf.reshape(net, [-1, net.shape.as_list()[1]*net.shape.as_list()[2]])
        print("flat shape", net.shape.as_list())
        for ind, units in enumerate(fc):
            net = tf.layers.dense(inputs=net, units=units, kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)
            activities[net.name] = net
            #net = tf.layers.dropout(net, rate=0.5)
            
            #print(scope.name + "shape", net.shape.as_list())
        ##tf.summary.histogram("dense_out", net)
        ##ipdb.set_trace()
        #variable_summaries(tf.trainable_variables()[-2])
        
            
    logits = tf.layers.dense(
                            inputs=net,
                            units=num_classes,
                            activation=tf.nn.softmax,
                            kernel_regularizer=regularizer,
                            name=scope.name)
    
    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    for var in all_trainable_vars:
        if 'kernel' in var.name:            
                kernels[var.name] = var
            
       
    return logits, kernels, activities, diversity




def CNN_Tutorial_Resi(x, output_channels=[32, 64, 128], seq_len=32, width=32, channels=3, pool_size=[2, 2], strides=[2, 2], filter_size=[3, 3], num_classes=10, fc=[500]):
    '''https://github.com/exelban/tensorflow-cifar-10/blob/master/include/model.py'''
    x = tf.reshape(x, [-1, seq_len, width, channels])   ###
    with tf.variable_scope('conv1') as scope:

        conv = tf.layers.conv2d(
            inputs=x,
            filters=output_channels[0],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        
        #conv = resBlock_CNN(conv, filter_size=filter_size[0], num_stacks=1, output_channels=output_channels[1], stride=[1, 1], name=scope.name, No_block=0)
        #conv = tf.layers.batch_normalization(conv)
        print(scope.name + "shape", conv.shape.as_list())
        #pool = tf.layers.max_pooling2d(resi_add, pool_size=pool_size, strides=strides, padding='SAME')
        #drop = tf.layers.dropout(pool, rate=0.25, name=scope.name)###0.25
        #print(scope.name + "shape", drop.shape.as_list())
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=output_channels[1],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        conv = resBlock_CNN(conv, filter_size=filter_size[0], num_stacks=1, output_channels=output_channels[2], stride=[1, 1], name=scope.name, No_block=1)
        #conv = tf.layers.batch_normalization(conv)
        print(scope.name + "shape", conv.shape.as_list())
        conv = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        conv = tf.layers.dropout(conv, rate=0.25, name=scope.name)###0.25
        print(scope.name + "shape", conv.shape.as_list())

        #### high-way net


    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=output_channels[2],
            kernel_size=filter_size[0],
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        #conv = resBlock_CNN(conv, filter_size=filter_size[0], num_stacks=1, output_channels=output_channels[2], stride=[1, 1], name=scope.name, No_block=2)
        print(scope.name + "shape", conv.shape.as_list())
        #conv = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        #print(scope.name + "shape", pool.shape.as_list())
        conv = tf.layers.conv2d(
            inputs=conv,
            filters=output_channels[2],
            kernel_size=filter_size[1],    #[2, 2],  #
            padding='SAME',
            kernel_regularizer=regularizer,
            activation=tf.nn.leaky_relu
        )
        conv = resBlock_CNN(conv, filter_size=filter_size[2], num_stacks=1, output_channels=output_channels[2], stride=[1, 1], name=scope.name, No_block=1)
        print(scope.name + "shape", conv.shape.as_list())
        conv = tf.layers.max_pooling2d(conv, pool_size=pool_size, strides=strides, padding='SAME')
        print(scope.name + "shape", conv.shape.as_list())
        conv = tf.layers.dropout(conv, rate=0.25, name=scope.name)   ###0.25
        ##### high-way net
        #H = tf.layers.dense(drop, units=num_outputs, activation=tf.nn.leaky_relu, name=scope.name+"H"))
        #T = tf.layers.dense(drop, units=num_outputs, activation=tf.nn.sigmoid, name=scope.name+"T")
        #C = 1. - T
        #drop = H * T + drop * C

    with tf.variable_scope('fully_connected') as scope:
        net = tf.reshape(conv, [-1, conv.shape[1]*conv.shape[2]*conv.shape[3]])
        print(scope.name + "shape", net.shape.as_list())
        ### dense layer
        for ind, units in enumerate(fc):
            net = tf.layers.dense(inputs=net, units=units, kernel_regularizer=regularizer, activation=tf.nn.leaky_relu)            
            net = tf.layers.dropout(net, rate=0.5)###0.50
            print(scope.name + "shape", net.shape.as_list())
        #tf.summary.histogram('pre_activation', net)

        logits = tf.layers.dense(
                                inputs=net,
                                units=num_classes,
                                activation=tf.nn.softmax,
                                kernel_regularizer=regularizer)
    ##### track all variables

    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


def DeepConvLSTM(x, output_channels=[8, 16, 32, 64], filter_size=[3, 3], pool_size=[2, 2], strides=[2, 2],  num_rnn=64, group_size=8, seq_len=10240, width=2, channels=1, fc=[1000], num_classes = 2):
    '''work is inspired by
    https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
    in-shape: (BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS), if no sliding, then it's the length of the sequence
    another from https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
    Param:
        output_channels: the number of filters to use 
        group_size: how many rows to group together to feed into LSTM
    '''

    net = tf.reshape(x,  [-1, seq_len,  width, channels])
    tf.summary.image('input_data', net, 4)
    print( net.shape.as_list())
    for layer_id, num_outputs in enumerate(output_channels):
        with tf.variable_scope("block_{}".format(layer_id)) as layer_scope:
            net = tf.layers.conv2d(
                                    inputs = net,
                                    filters = num_outputs,
                                    kernel_size = filter_size,
                                    padding = 'same',
                                    activation=tf.nn.leaky_relu
                                    )
            print(net.name, net.shape.as_list())
            tf.summary.histogram(net.name, net)
            #net = tf.layers.dropout(net, rate=0.25)

            net = tf.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
            net = tf.layers.dropout(net, rate=0.4)
            
            print("before LSTM", net.shape.as_list())

    with tf.variable_scope("reshape4rnn") as layer_scope:
        ### prepare input data for rnn requirements. current shape=[None, seq_len, output_channels]
        ### Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
        #ipdb.set_trace()
        net = tf.reshape(net, [-1, net.shape[1]//group_size, net.shape[2]*net.shape[-1]*group_size])   ## group these data points together 
        print("net ", net.shape.as_list())
        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        net = tf.unstack(net, axis=1)

        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_rnn)
        outputs, hid_states = tf.nn.static_rnn(lstm_layer, net, dtype=tf.float32)  ###outputsoutputsA 2-D tensor with shape [batch_size, self.output_size].

        #track LSTM histogram
        #for one_lstm_cell in lstm_layer:
            #one_kernel, one_bias = one_lstm_cell.variables
            ## I think TensorBoard handles summaries with the same name fine.
            #tf.summary.histogram("LSTM-Kernel", one_kernel)
            #tf.summary.histogram("LSTM-Bias", one_bias)
            
    with tf.variable_scope("dense_out") as layer_scope:
        #ipdb.set_trace()
        #net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        net  = tf.reshape(outputs[-1], [-1, num_rnn])
        print("reshape net ", net.shape.as_list())

        ### dense layer
        for ind, units in enumerate(fc):
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            
        tf.summary.histogram('pre_activation', net)
        print("net ", net.shape.as_list())
        ### logits
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net ", logits.shape.as_list())
        tf.summary.histogram('logits', net)

    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var

    return logits, kernels


def RNN(x, num_rnn=128, seq_len=10240, width=2, channels=1, group_size=32, fc=[200], num_classes = 2):
    '''Use RNN
    x: shape[batch_size,time_steps,n_input]'''
    with tf.variable_scope("rnn_lstm") as layer_scope:
        net = tf.reshape(x, [-1, seq_len, width, channels])
        #ipdb.set_trace()
        #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
        net = tf.reshape(net, [-1, seq_len//group_size, group_size*2])   ### feed not only one row but 8 rows of raw data
        print("net ", net.shape.as_list())
        net = tf.unstack(net, axis=1)

        ##### defining the network
        lstm_layer = tf.contrib.rnn.BasicLSTMCell(num_rnn, forget_bias=1)
        outputs, _ =tf.nn.static_rnn(lstm_layer, net, dtype="float32")
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        print("net ", net.shape.as_list())
        
        for ind, units in enumerate(fc):
            net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
        tf.summary.histogram('pre_activation', net)
        
        net = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
        print("net", net.shape.as_list())
        tf.summary.histogram('logits', net)
    return net

def RNN_Tutorial(x, num_rnn=[100, 100], seq_len=10240, width=2, num_seg=119, channels=1, fc=[100, 100], group_size=1, drop_rate=0.25, num_classes = 2):
    '''based on the method in https://ieeexplore.ieee.org/document/7727334/
    1. from the correlation length dixtribution get the optimal segment length--86, then 10240 // 86 = 119-- majority vote
    '''
    #with tf.variable_scope("rnn_lstm") as layer_scope:
    #net = tf.reshape(x, [-1, seq_len, width, channels])
    #### prepare the shape for rnn: "time_steps" number of [batch_size,n_input] tensors
    #assert( seq_len % group_size == 0, 'The seq_len should int divide the group size')
    net = tf.reshape(x, [-1, seq_len//group_size, width*channels*group_size])   ### feed not only one row but 8 rows of raw data
    print("reshape input", net.shape)
    ##### defining the network
    with tf.variable_scope("lstm_fc0") as scope:
        #ipdb.set_trace()
        net = tf.unstack(net, axis=1)    ##<tf.Tensor 'lstm_fc/unstack_1:num_seg' shape=(?, 2) dtype=float32>]
        rnn_layer = tf.contrib.rnn.GRUCell(num_rnn[0], kernel_initializer=initializer)
        outputs, _ =tf.nn.static_rnn(rnn_layer, net, dtype="float32", scope=scope)
        #print("lstm {} out".format(ind), outputs.shape)
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        #ipdb.set_trace()
        tf.summary.histogram('pre_activation', net)
        net = tf.layers.dense(inputs=net, units=fc[0], activation=None)
        #net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(inputs=net, rate=drop_rate)
        print("lstm 0 out", net.shape)
        tf.summary.histogram('pre_activation', net)
        net = tf.reshape(net, [-1, net.shape[1]//group_size, group_size])   ## reshape to a sequece for RNN

    with tf.variable_scope("lstm_fc1") as scope:
        net = tf.unstack(net, axis=1)    ##<tf.Tensor 'lstm_fc/unstack_1:num_seg' shape=(?, 2) dtype=float32>]
        rnn_layer = tf.contrib.rnn.GRUCell(num_rnn[1])
        outputs, _ =tf.nn.static_rnn(rnn_layer, net, dtype="float32", scope=scope)
        net = tf.layers.batch_normalization(outputs[-1], center = True, scale = True)
        print("lstm 1 out", net.shape)
        tf.summary.histogram('logits', net)
        net = tf.layers.dropout(inputs=net, rate=drop_rate)
        logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)
    print("logits", logits.shape.as_list())
    
    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels
    
    

def Atrous_CNN(x, output_channels_cnn=[8, 16, 32, 64], dilation_rate=[2, 4, 8, 16], filter_size = [9, 1], pool_size=[2, 2], strides=[2, 2], fc=[200],seq_len=10240, width=1, num_classes = 10):
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
    image_level_features = tf.reduce_mean(net, pool_size, name='image_level_global_pool', keep_dims=True)
    print("image_level_features", image_level_features.shape.as_list())
    image_level_features = tf.layers.conv2d(
                                            inputs = image_level_features,
                                            filters = output_channels_cnn,
                                            kernel_size=[1, 1],
                                            activation=tf.nn.leaky_relu)
    print("image_level_features", image_level_features.shape.as_list())
    #ipdb.set_trace()
    net = tf.layers.max_pooling2d(inputs=image_level_features, pool_size=pool_size, strides=strides)
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
                                                kernel_size=filter_size,
                                                padding = 'same',
                                                activation=tf.nn.leaky_relu)
            conv_net = tf.layers.max_pooling2d(inputs=conv_net, pool_size=pool_size, strides=strides)
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
                                 dilation_rate = dilation_rate[jj],
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
                             activation = tf.nn.leaky_relu)
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
    for ind, units in enumerate(fc):
        net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net)

    print("net ", net.shape.as_list())
    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    ''''''
    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


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
        H = tf.layers.dense(net, units=num_outputs, activation=tf.nn.leaky_relu, name="denseH{}".format(jj))
        T = tf.layers.dense(net, units=num_outputs, activation=tf.nn.sigmoid, name="denseT{}".format(jj))
        C = 1. - T
        net = H * T + net * C
    print("net", net.shape.as_list())

    net = tf.layers.batch_normalization(net, center = True, scale = True)
    
    ### all the segments comes back together
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    
    print("net unite", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=200, activation=tf.nn.leaky_relu)
    net = tf.layers.batch_normalization(net, center = True, scale = True)
    net = tf.layers.dense(inputs=net, units=50, activation=tf.nn.leaky_relu)
  #net = tf.layers.dropout(inputs=net, rate=0.75)

    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)

    return logits

def PyramidPoolingConv(x, output_channels=[2, 4, 8, 16, 32, 64, 128], filter_size=[[5, 1]], dilation_rate=[[2,1], [8,1], [16,1], [32,1]], fc=[500], strides=[2, 1], seq_len=10240, width=2, channels=1, num_seg=5, num_classes=2):
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
                                         dilation_rate = rate,
                                         padding = 'same',
                                         activation = None)
        net = tf.layers.batch_normalization(net, center = True, scale = True)
        net = tf.nn.leaky_relu(net)
        print("input net", net.shape.as_list())
        for ind, num_filter in enumerate( output_channels):
            ### start conv with each level dilation 
            with tf.variable_scope("dilate{}_conv{}".format(jj, ind)) as layer_scope:

                net = tf.layers.conv2d(
                                         inputs = net,
                                         filters = num_filter,   ##[filter_height, filter_width, in_channels, out_channels]
                                         kernel_size = filter_size,
                                         strides = strides,
                                         padding = 'same',
                                         activation = tf.nn.leaky_relu)
                net = tf.layers.batch_normalization(net, center = True, scale = True)
                #### high-way net
                H = tf.layers.dense(net, units=num_filter, activation=tf.nn.leaky_relu, name="denseH{}".format(ind))
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
                                         activation = tf.nn.leaky_relu)
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
                             activation = tf.nn.leaky_relu)
    print("last last net", net.shape.as_list())

    '''########### Dense layer ##################3'''
    #net = tf.layers.flatten(net, [-1, ] )
    net = tf.reshape(net, [-1,  net.shape[1]*net.shape[2]*net.shape[3]*num_seg])
    print("flatten net ", net.shape.as_list())

    ## Logits layer
    for ind, units in enumerate(fc):
        net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net)
        
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
                            activation = tf.nn.leaky_relu)
    print("net1x1", net_1x1.shape.as_list())
    filter_concat.append(net_1x1)
    
    ## branch No.2/3, 5*1, 3*1 conv level
    for ind, kernel in enumerate(filter_size):
        net = tf.layers.conv2d(
                            inputs = x,
                            filters = reduce_chanenls, 
                            kernel_size = [1, 1],
                            padding = 'same',
                            activation = tf.nn.leaky_relu)
        print("net1x1 in reduce", net.shape.as_list())
        net = tf.layers.conv2d(
                            inputs = net,
                            filters = out_channels, 
                            kernel_size = kernel,   ### seq: 1
                            padding = 'same',
                            activation = tf.nn.leaky_relu)
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
                        activation = tf.nn.leaky_relu)
                        
    print("net reduce pooling- pooliing", net.shape.as_list())
    filter_concat.append(net)

    #### concat all feature maps from 4 branches
    inception = tf.nn.leaky_relu(tf.concat(([filter_concat[i] for i in range( len(filter_concat))]), axis=3,name="concat"))
    print("inception concat", inception.shape.as_list())

    return inception


    
def Inception_complex(x, output_channels=[16, 32], filter_size=[[5, 1], [9, 1]], reduce_chanenls=16, pool_size=[2, 1], strides=[2, 1], fc=[500], num_blocks=2, seq_len=10240, width=2, channels=1, num_classes=2):
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
    for ind, units in enumerate(fc):
        net = tf.layers.dense(inputs=net, units=units, activation=tf.nn.leaky_relu)
        net = tf.layers.batch_normalization(net)

    ## Logits layer
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.sigmoid)
    
    
    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels


def ResNet(x, num_layer_per_block=3, filter_size=[[5, 1], [9, 1]], pool_size=[[2, 1]], strides=[2, 1], output_channels=[32, 64, 128], fc=[500], seq_len=10240, width=2, channels=1, num_classes=2):
    '''https://medium.com/@pierre_guillou/understand-how-works-resnet-without-talking-about-residual-64698f157e0c
    Within one block the num of filters is the same, only when downsampling the num of filter will increase.
    within one block, there is several stacks, one stack usually have two conv layers
    34-layer-residual structure
    ('1stlayershape', [None, 4750, 2, 16])
    ('block0shape', [None, 1187, 2, 16])
    ('block0/block0_stack0-out', [None, 1187, 2, 16])
    ('block0/block0_stack1-out', [None, 1187, 2, 16])
    ('block0/block0_stack2-out', [None, 1187, 2, 16])
    ('block1shape', [None, 296, 2, 32])
    ('block1/block1_stack0-out', [None, 296, 2, 32])
    ('block1/block1_stack1-out', [None, 296, 2, 32])
    ('block1/block1_stack2-out', [None, 296, 2, 32])
    ('block2shape', [None, 74, 2, 64])
    ('block2/block2_stack0-out', [None, 74, 2, 64])
    ('block2/block2_stack1-out', [None, 74, 2, 64])
    ('block2/block2_stack2-out', [None, 74, 2, 64])
    ('average_poolshape', [None, 37, 2, 64])
    ('average_poolshape', [None, 37, 2, 64])
    ('flatten shape', [None, 4736])
    ('denseshape', [None, 500])
    '''
    with tf.variable_scope("1stlayer") as scope:
        net = tf.reshape(x, [-1, seq_len, width, channels])
        net = tf.layers.conv2d( 
                                inputs = net,
                                filters = output_channels[0],
                                kernel_size = filter_size[0],
                                padding = 'same',
                                activation = None)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
        net = tf.nn.leaky_relu(net)
        print(scope.name + "shape", net.shape.as_list())
   
    for block in range(len(output_channels)):
        with tf.variable_scope("block{}".format(block)) as scope:
            ### downsample         
            net = tf.layers.conv2d( 
                                    inputs = net,
                                    filters = output_channels[block],
                                    kernel_size = filter_size[1],
                                    padding = 'same',
                                    activation = None)
            net = tf.layers.max_pooling2d(inputs=net, pool_size=pool_size[0], strides=strides)
            net = tf.layers.batch_normalization(net)
            net = tf.nn.leaky_relu(net)
            print(scope.name + "shape", net.shape.as_list())
            ### large residual block with specified stacks
            net = resBlock_CNN(net, filter_size=filter_size[1], num_stacks=num_layer_per_block, output_channels=output_channels[block], stride=[1, 1], name=scope.name)
        
    ### average pool
    with tf.variable_scope("average_pool") as scope:
        net = tf.layers.average_pooling2d(inputs=net, pool_size=[2, 1], strides=[2, 1])
        print(scope.name + "shape", net.shape.as_list())
    print(scope.name + "shape", net.shape.as_list())
    net = tf.reshape(net, [-1, net.shape[1]*net.shape[2]*net.shape[3]])
    print("flatten shape", net.shape.as_list())
    ### dense layer
    with tf.variable_scope("dense") as scope:
        for ind, units in enumerate(fc):
            net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu)
            net = tf.layers.batch_normalization(net)
            net = tf.layers.dropout(net, rate=0.25, name=scope.name) ##dropout rate,
            print(scope.name + "shape", net.shape.as_list())
    logits = tf.layers.dense(net, units=num_classes, activation=tf.nn.softmax)

    ##### track all variables
    kernels = {}
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in all_trainable_vars:
        if 'kernel' in var.name:            
                kernels[var.name] = var
                
    return logits, kernels
   

def Bottleneck_stage(x, out_channel, num_stack=3, filter_size=[3, 3], strides=[2, 2], cardinality=8, name=0):
    '''one Aggregate block'''
    in_channel = x.shape[-1]
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
                                        activation = tf.nn.leaky_relu
                                    )
                net = tf.layers.batch_normalization(net)
                #net = tf.nn.leaky_relu(net)
                
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = 4,
                                        kernel_size = filter_size,
                                        padding = 'same',
                                        activation = tf.nn.leaky_relu
                                        )
                net = tf.layers.batch_normalization(net)
                #net = tf.nn.leaky_relu(net)
                
                net = tf.layers.conv2d(
                                        inputs = net,
                                        filters = in_channel,
                                        kernel_size = [1, 1],
                                        padding = 'same',
                                        activation = None
                                        )
                aggregate_net += net
            
            aggregate_net = tf.layers.batch_normalization(aggregate_net)
            aggregate_net = tf.nn.leaky_relu(aggregate_net)
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
        net = tf.nn.leaky_relu(net)
        print("resi{}_output {}".format(name, stack), inputs.shape.as_list())

        return net
            

def AggResNet(x, output_channels=[2, 4, 8], num_stacks=[3, 4, 3], cardinality=16, seq_len=10240, width=2, channels=1, filter_size=[[11, 1], [9, 1]], pool_size=[2, 1], strides=[2, 1], fc=[500], num_classes=2, ifaverage_pool=False):
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
                            activation = tf.nn.leaky_relu
                            )
    net = tf.layers.batch_normalization(net)
    #net = tf.layers.dropout(inputs=net, rate=0.2)
    #net = tf.nn.leaky_relu(net)
    print("starting 3x3 conv", net.shape.as_list())

        
    for ind, out_channel in enumerate(output_channels):
        print("Stage {} start".format(ind), net.shape.as_list())
        
        net = Bottleneck_stage(net, out_channel, num_stack=num_stacks[ind], cardinality=cardinality, name=ind)
    #ipdb.set_trace()    
    net = tf.layers.average_pooling2d(inputs=net, pool_size=pool_size, strides=strides, padding='same')
    print("pooling", net.shape.as_list())
    net = tf.layers.flatten(net)
    print("flatten", net.shape.as_list())
    net = tf.layers.dense(inputs=net, units=fc[0], activation=tf.nn.leaky_relu)
    print("dense1", net.shape.as_list())
    net = tf.layers.batch_normalization(net)
    #net = tf.layers.dropout(inputs=net, rate=0.2)
    logits = tf.layers.dense(inputs=net, units=num_classes, activation=tf.nn.softmax)

    ##### track all variables
    all_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    kernels = {}
    for var in all_trainable_vars:
        if 'kernel' in var.name:
            kernels[var.name] = var
            
    return logits, kernels
        

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

        # leaky_relu is performed right after each batch normalization,
        # expect for the output of the block where leaky_relu is performed after the adding to the shortcut
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
    ##e(n,k,l) = [w'(s,k)]T*leaky_relu(W(s,k)f(n,l) + b(s,k)) + b'(s,k),
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
