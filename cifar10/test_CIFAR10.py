 ### use basic network to do classification
 ##https://github.com/exelban/tensorflow-cifar-10/blob/master/train.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
#from statsmodels.tsa.seasonal import seasonal_decompose
import functions as func
import modules as mod
from keras.datasets import cifar10
import ipdb

'''batch_size=100, adam, lr=0.001'''


def lr(epoch):
    learning_rate = 0.001
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 40:
        learning_rate *= 1e-3
    elif epoch > 20:
        learning_rate *= 1e-2
    elif epoch > 10:
        learning_rate *= 1e-1
    return learning_rate

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
plot_every = 500
save_every = 1
height, width, channels = 32, 32, 3 #seq_len, 1     # MNIST
batch_size = 128 # old: 16     20has a very good result
num_classes = 10
#pattern='ds_8*.csv'
version = 'CNN_tutorial'  #'CNN_tutorial' #'AggResNet'  #'Resi_HighwayFNN'  #'Plain_CNN'   ## 'CNN'  ###'Inception'   #            #DilatedCNNDeepCLSTM'whole_{}_DeepCLSTM'.format(pattern[0:4])       #### DeepConvLSTMDeepCLSTM
data_dir = 'cifar_data/cifar10/'
results_dir= "results/3-CIFAR10_checks/" + version + '/batch{}/' .format(batch_size)+ datetime
logdir = results_dir+ "/model"


'''FNN data shape'''
x = tf.placeholder("float32", [None, height, width, channels], 'input_images')  #20s recording width of each recording is 1, there are 2 channels
x_image = tf.reshape(x, [-1, height, width, channels], 'input_images')  #20s recording width of each recording is 1, there are 2 channels
y = tf.placeholder("float32", [None, num_classes], 'labels')
learning_rate = tf.placeholder("float32")

### load data
(data_train, y_train), (data_test, y_test) = cifar10.load_data()
## x_train(50000, 32, 32, 3)
## test(10000, 32, 32, 3)
data_train = data_train / 255.0
data_test = data_test / 255.0
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

epochs = 251
total_batches =  len(data_train) // (20* batch_size) #5001               #
print('num_train', len(data_train), 'total batches', total_batches)

def evaluate_on_test(sess, epoch, accuracy, cost, outputs, test_data, kernels, save_name='results/'):

    x_test, y_test = test_data
    test_labels = np.eye(num_classes)[y_test]

    acc_epoch_test = 0
    loss_epoch_test = 0
    ### test on the whole test set
    for k in range(len(x_test) // batch_size):
        test_batch = x_test[k*batch_size : (k+1)*batch_size]
        test_labels = np.eye(num_classes)[y_test[k*batch_size : (k+1)*batch_size]]

        test_acc, test_cost, train_vars, out_logits = sess.run([accuracy, cost, kernels, outputs], {x: test_batch, y: test_labels, learning_rate:lr(epoch)})   # test_acc_sum, sensitivity_sum, specificity_sum,
        acc_epoch_test += test_acc
        loss_epoch_test += test_cost

        #labels = np.argmax(out_logits)
    acc_epoch_test /= (k + 1)
    loss_epoch_test /= (k + 1)

    #ipdb.set_trace()
    if epoch % 3 == 0:
        for ind, var in enumerate(train_vars):
            #if 'fully' in var:
                #if train_vars[var].shape[-1] > num_classes:
                    #plt.imshow(train_vars[var], cmap='viridis', aspect='auto')
                    #plt.title(var + '-' + np.str(train_vars[var].shape))
                    #plt.ylabel('in unit index')
                    #plt.xlabel('out unit index')
                    #plt.colorbar()
                    #plt.savefig(save_name + '/fully-hid-' + np.str(train_vars[var].shape)+'-epoch-{}.png'.format(epoch), format='png')
                    #plt.close()
                #else:
                    
                    #for ind in range(train_vars[var].shape[-1]):
                        #plt.plot(train_vars[var][:, ind], label='label-{}'.format(ind))
                    ##plt.imshow(train_vars[var], cmap='viridis', aspect='auto')
                    ##plt.legend(loc='best')
                    #plt.ylabel('in unit index')
                    #plt.xlabel('out unit index')
                    #plt.savefig(save_name + '/fully-logits-imshow' + np.str(train_vars[var].shape)+'-epoch-{}.png'.format(epoch), format='png')
                    #plt.close()
            if 'conv' in var:
                ipdb.set_trace()
                func.put_kernels_on_grid(train_vars[var], pad=1, save_name=save_name+'/conv_kernel', mode='imshow')
    #for ind, net in enumerate(acti):
        #if 'conv' in net:
            #func.plot_conved_image(net, save_name=save_name+'/'+net[0:5])
            

    
    return acc_epoch_test, loss_epoch_test


    

### construct the network
def train(x):

    #### Constructing the network
    #outputs, kernels = mod.fc_net(x, hid_dims=[500, 300], num_classes = num_classes)   ##
    #outputs, kernels = mod.resi_net(x, hid_dims=[500, 300], seq_len=height, width=width, channels=channels, num_blocks=2, num_classes = num_classes)   ## works, not amazing
    #outputs, kernels = mod.CNN(x, output_channels=[16, 32, 64], num_block=3, filter_size=[3, 3], strides=[2, 2], seq_len=height, width=width, channels=channels, num_classes = num_classes)    ## ok
    #outputs, kernels = mod.Plain_CNN(x, output_channels=[32, 64, 128], num_block=3, pool_size=[2, 2], filter_size=[3, 3], strides=[2, 2], seq_len=height, width=width, channels=channels, num_classes = num_classes)    ## ok
    
    #outputs, kernels = mod.DeepConvLSTM(x, output_channels=[8, 16, 32], filter_size=[3, 3], num_lstm=64, pool_size=[2, 2], strides=[2, 2],  group_size=1, seq_len=height, width=width, channels=channels, num_classes = num_classes)  ## ok
    #outputs, kernels = mod.RNN(x, num_lstm=64, seq_len=height, width=width, num_classes = num_classes)   ##ok
    #outputs, kernels = mod.Atrous_CNN(x, output_channels_cnn=[4, 8, 16], dilation_rate=[[2,2], [4, 4], [8, 8]], kernel_size = [3, 3], seq_len=height, width=width, channels=channels, num_classes = num_classes) ##ok
    #outputs, kernels = mod.Inception(x, filter_size=[[3, 3], [5, 5]],num_block=2, seq_len=height, width=width, channels=channels, num_classes=num_classes)

    #outputs, kernels = mod.Inception_complex(x, output_channels=[16, 32], filter_size=[[3, 3], [5, 5]], pool_size=[2, 2], strides=[2, 2], num_blocks=2, seq_len=height, width=width, channels=channels, num_classes=num_classes)
    #outputs, kernels = mod.AggResNet(x, output_channels=[16, 32, 64], num_stacks=[3, 3, 3], cardinality=16, seq_len=height, width=width, channels=channels, filter_size=[[3, 3], [2, 2]], pool_size=[2, 2], strides=[2, 2], fc1=200, num_classes=num_classes)   ## output_channels should be the same length as num_subBlocks
    outputs, kernels = mod.CNN_Tutorial(x, output_channels=[32, 64, 64], seq_len=height, width=width, channels=channels, pool_size=[2, 2], strides=[2, 2], filter_size=[[3, 3], [2, 2]], num_classes=num_classes, fc=[500])### this works very well on both
    #outputs, kernels = mod.CNN_Tutorial_Resi(x, output_channels=[16, 32, 64], seq_len=height, width=width, channels=channels, pool_size=[2, 2], strides=[2, 2], filter_size=[[3, 3], [2, 2]], num_classes=num_classes, fc1=200)###
    #ipdb.set_trace()

    with tf.name_scope("loss"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y), name="cost")
    with tf.name_scope("performance"):
        predictions = tf.argmax(outputs, 1)
        correct = tf.equal(predictions, tf.argmax(y, 1), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, "float32"), name="accuracy")


        tf.summary.scalar('loss', cost)
        tf.summary.scalar('accuracy', accuracy)

    optimizer = tf.train.AdamOptimizer(
                            learning_rate=learning_rate,
                            beta1=0.9,  ##The exponential decay rate for the 1st moment estimates.
                            beta2=0.999,  ##  The exponential decay rate for the 2nd moment estimates
                            epsilon=1e-08).minimize(cost)
    #optimizer = tf.train.RMSPropOptimizer(
                            #learning_rate=learning_rate,
                            #momentum=0.9, 
                            #epsilon=1e-08).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    #################### Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata()
    summaries = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep= 20)

    with tf.Session() as sess:
         #profiler = tf.profiler.Profiler(sess.graph)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        print(results_dir)

        np.random.seed(1998745)
        sess.run(tf.global_variables_initializer())
        ## tracking performance for all the traiing epochs
        acc_total_train = []
        acc_total_test = []
        loss_total_train = []
        loss_total_test = []
        for epoch in range(epochs):
            ### tracking within one epoch
            acc_epoch_train = 0
            loss_epoch_train = 0         
                        
            for batch in range(total_batches):###
                save_name = results_dir + '/' + "training_samples_epoch{}_batch{}_".format(epoch, batch)

                ########## Data
                batch_data = data_train[batch*batch_size : (batch+1)*batch_size]
                batch_labels = np.eye(num_classes)[y_train[batch*batch_size : (batch+1)*batch_size]]

                _, acc, c = sess.run([optimizer, accuracy, cost], feed_dict={x: batch_data, y: batch_labels, learning_rate:lr(epoch)})

                acc_epoch_train += acc
                loss_epoch_train += c
                ### Test
                #if batch % (total_batches//3) == 0:
                    #acc_epoch_test = 0
                    #loss_epoch_test = 0
                    #### test on the whole test set
                    #(data_test, y_test)
                    #for k in range(len(data_test) // batch_size):
                        #test_data = data_test[k*batch_size : (k+1)*batch_size]
                        #test_labels = np.eye(num_classes)[y_test[k*batch_size : (k+1)*batch_size]]

                        #test_acc, test_cost = sess.run([accuracy, cost], {x: test_data, y: test_labels, learning_rate:lr(epoch)})   # test_acc_sum, sensitivity_sum, specificity_sum,
                        #acc_epoch_test += test_acc
                        #loss_epoch_test += test_cost
                        
                    #acc_epoch_test = acc_epoch_test/ (k + 1)
                    #loss_epoch_test = loss_epoch_test/ (k + 1)
                if batch % 100 == 0:
                    summary = sess.run(summaries, feed_dict={x: batch_data, y: batch_labels, learning_rate:lr(epoch)})
                    print("epoch", epoch, "batch", batch, 'loss', c, 'train_accuracy', acc)
                    #######################################################
                    
                    writer.add_summary(summary, epoch*total_batches+batch)
                
            if epoch % 1 == 0:                
                acc_test, loss_test = evaluate_on_test(sess, epoch, accuracy, cost, outputs, (data_test, y_test), kernels, save_name=results_dir)
                print("epoch", epoch, "batch", batch, 'loss', c, 'train_accuracy', acc, 'test_acc', acc_test)
                
            acc_epoch_train = acc_epoch_train / total_batches * 1.0
            loss_epoch_train = loss_epoch_train / total_batches * 1.0

            if epoch % save_every == 0:
                func.save_model(saver, sess, logdir, epoch)
                last_saved_step = epoch

            # track training and testing
            loss_total_train.append(loss_epoch_train)            
            acc_total_train.append(acc_epoch_train)
            loss_total_test.append(loss_test)            
            acc_total_test.append(acc_test)

            if epoch % 1 == 0:
                if epoch < 2 :
                    rand_int = np.random.choice(len(data_train), 20)
                    
                    func.plot_train_samples(data_train[rand_int], y_train[rand_int], ylabel=None, save_name=save_name)
                print("loss_total_train", loss_total_train, "loss_total_test", loss_total_test)
                func.plot_smooth_shadow_curve([acc_total_train, acc_total_test], hlines=[0.7, 0.8, 0.85], xlabel= 'training epochs', ifsmooth=False, ylabel="accuracy", colors=['darkcyan', 'royalblue'], title='Learing curve', labels=['accuracy_train', 'accuracy_test'], save_name=results_dir+ "/learning_curve_epoch_{}".format(epoch))

                func.plot_smooth_shadow_curve([loss_total_train, loss_total_test], colors=['c', 'm'], xlabel= 'training epochs',ifsmooth=False, ylabel="loss", title='Loss',labels=['training', 'testing'], save_name=results_dir+ "/losses_epoch_{}".format(epoch))

                func.save_data_to_csv((acc_total_train, loss_total_train, acc_total_test, loss_total_test), header='accuracy_train,loss_train,accuracy_test,loss_test', save_name=results_dir + '/' +'epoch_accuracy.csv') 
        

        


if __name__ == "__main__":
    train(x)
### define the cost and opti
