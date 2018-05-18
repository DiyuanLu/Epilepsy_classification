import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.models import Model
from keras import objectives
#from keras.datasets import mnist
from keras import backend as K
import datetime
import ipdb
from keras.models import model_from_json
import functions as func
import random
import pickle



def sampling(args):
    '''# a keras lambda layer computes arbitrary function on the output of a layer
# so z is effectively combining mean and variance layers through sampling func 
'''
    _mean, _log_var = args
    epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0., stddev = epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon

random.seed(1998)

batch_size =  20            # train: 5955    test:  1485 samples
ifaverage = False                    ###False   ###
if ifaverage:
    channel = 1
else:
    channel  = 2
original_dim = 1280###10240#128
window = 128
seq_shape = (original_dim, channel)
intermediate_dim = 64
latent_dim = 2

nb_epochs = 50
epsilon_std = 1.0
data_dir = "data/npy_data/sub700-norm0~1.npz" #testF751testN1501_aver_zscoreds_8testF751testN1501_averageds_8testF751testN1501_average

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir= "results/vae-keras-ori-eeg-norm/".format(batch_size)+ datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print results_dir

#sess = K.get_session()
#sess = tfdb.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)
'''################## Encoder ##################'''
X = Input(batch_shape = (batch_size, original_dim*channel))
net  =  Dense(intermediate_dim, activation = 'relu')(X)
z_mean = Dense(latent_dim)(net)
z_log_var = Dense(latent_dim)(net)
z = Lambda(sampling, output_shape = (latent_dim,))([z_mean, z_log_var])

'''#################### Decoder graph #########################'''
h_decoder = Dense(intermediate_dim, activation = 'relu')
X_bar = Dense(original_dim * channel, activation = 'sigmoid')
# we instantiate these layers separately so as to reuse them later
dec_net = h_decoder(z)
X_reconstruction  = X_bar(dec_net )

'''############ loss ######################3'''
def vae_loss(x, x_rec):
    reconst_loss = objectives.binary_crossentropy(x, x_rec)
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    return reconst_loss + kl_loss
# build and compile model
vae = Model(X , X_reconstruction)
vae.compile(optimizer = 'adam', loss = vae_loss)

# load data for training
data_file = np.load(data_dir)
x_train, y_train, x_test, y_test = data_file["x_train"], data_file["y_train"], data_file["x_test"], data_file["y_test"]
ipdb.set_trace()
#####plot samples and predict label
num_sample = 20
sample_ind = np.random.choice(100, [num_sample])
sample_img = x_test[sample_ind, :, :]   ### 10240 * 2
sample_label = y_test[sample_ind]   ### 
#samples =sample_img.reshape(-1, window)
plt.figure()
for ii in range(num_sample):
    ax1 = plt.subplot(5, 4, ii +1)  
    plt.plot(sample_img[ii, :, 0])
    plt.xlabel("label: "+ np.str(sample_label[ii]))
    plt.xlim([0, original_dim])
    #plt.setp(ax1.get_yticklabels(), visible = False)
    plt.setp(ax1.get_xticklabels(), visible = False)
plt.tight_layout()
plt.savefig(results_dir + '/sampled_test.png', format = 'png')
plt.close()


# normalize input and make them float32 to run on GPU
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert 28x28 images into 784-vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# x_train is required for input and loss output as target
history = vae.fit(x_train, x_train, shuffle = True, epochs = nb_epochs, batch_size = batch_size, validation_data = (x_test, x_test))

with open(results_dir + '/trainHIstoryDIct', 'wb') as filepi:
    pickle.dump(history.history, filepi)
############### encoder is the inference network
encoder = Model(X, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size = batch_size)
plt.figure(figsize = (6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c = y_test)
plt.colorbar()
plt.savefig(results_dir + '/latent_scatter.png', format='png')
plt.close()


# since the generator treats z as an input, we make z an input layer
z_input = Input(shape = (latent_dim,))
_h_decoded = h_decoder(z_input)
_x_decoded = X_bar(_h_decoded)
generator =  Model(z_input, _x_decoded)
#####plot sample reconstruction
num_sample = 10
z_sample = np.random.normal(0, 1, (num_sample, latent_dim))
print "z_sample.shape", z_sample.shape
x_decoded = generator.predict(z_sample)      ### start from random
x_test_recon = generator.predict(x_test_encoded[0:num_sample, :])      # use the encoding from test reconstruct
x_test_recon = x_test_recon.reshape(-1, original_dim, channel )      # batch_size, 20480 -- batch_size, 10240, 2
sampled_im = x_decoded.reshape(-1, original_dim, channel )   ## batch_size, 20480 -- batch_size, 10240, 2
plt.figure()
for ii in range(num_sample):
    ax1 = plt.subplot(5, 2, ii +1)  
    plt.plot(sampled_im[ii, :, 0])
    plt.xlim([0, original_dim])
    plt.xlabel('z:{}'.format(z_sample[ii, :]))
plt.tight_layout()
plt.savefig(results_dir + '/sampled_reconstructon.png', format='png')
plt.close()


plt.figure()
x_test_reshape = x_test[sample_ind, :].reshape(-1, original_dim, channel )
y_test_label = y_test[sample_ind]
for ii in range(5):
    ax1 = plt.subplot(5,2, ii *2 +1)
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.xlabel("z:{}".format(x_test_encoded[0:num_sample, :][ii]))
    plt.plot(x_test_recon[ii, :, 0])
    plt.xlim([0, original_dim])
    ax2 = plt.subplot(5,2, (ii+1) *2 )  
    plt.plot(x_test_reshape[ii, :, 0])
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.xlabel("label:{}".format(np.str(y_test_label[ii])))
    plt.xlim([0, original_dim])
    if ii == 0:
        ax1.set_title("reconstruction")
        ax2.set_title("original")
plt.savefig(results_dir + '/test_reconstructons.png', format='png')
plt.close()

#####plot prior
# 2d manifold of images by exploring quantiles of normal dist (using the inverse of cdf)

num = 5
figure  =  plt.subplot()
grid_x = np.linspace(0.05, 0.95, num)##norm.ppf(np.linspace(0.05, 0.95, num))
grid_y = np.linspace(0.05, 0.95, num)
#latent = np.random.normal(0, 1, latent_dim)
f, axarr = plt.subplots(num, num, sharex='col', sharey='row')

for ii, yi in enumerate(grid_y):
    for jj,xi in enumerate(grid_x):
        latent =  np.array([[xi, yi]]) 
        x_decoded = generator.predict(latent)
        x_decoded = x_decoded.reshape(-1, original_dim, channel )   ## x_decoded.shape = (1, 10240, 2)
        axarr[ii, jj].plot(x_decoded[0, :, 0], 'royalblue')
        axarr[ii, jj].set_xlim(0, original_dim)
ipdb.set_trace()
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.savefig(results_dir + '/prior_{}*{}.png'.format(num, num), format='png')
plt.close()
ipdb.set_trace()

# serialize model to JSON
model_json = vae.to_json()
with open(logdir + "/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
vae.save_weights(logdir + "/model.h5")
print("Saved model to disk")
# later...
## load json and create model
json_file = open(logdir+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, {'batch_size': batch_size, 'latent_dim': latent_dim, 'epsilon_std':epsilon_std})  ### have to give the hyperparams

# load weights into new model
loaded_model.load_weights(logdir+"/model.h5")
print("Loaded model from disk")
 
## evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, x_test, batch_size=batch_size)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
'''
Epoch 1/20
5955/5955 [==============================] - 6s 950us/step - loss: 40622.3800 - val_loss: 2549816.5934
Epoch 5/20
5955/5955 [==============================] - 5s 911us/step - loss: 9.0785 - val_loss: 167811.6405
Epoch 15/20
5955/5955 [==============================] - 5s 896us/step - loss: 6.6307 - val_loss: 116943.6244
Epoch 20/20
5955/5955 [==============================] - 5s 918us/step - loss: 6.1745 - val_loss: 145969.5441
'''
'''
Epoch 1/30
5955/5955 [==============================] - 4s 728us/step - loss: 71.0389 - val_loss: 2.5578
Epoch 2/30
5955/5955 [==============================] - 4s 710us/step - loss: 2.7994 - val_loss: 3.1707
Epoch 3/30
5955/5955 [==============================] - 4s 709us/step - loss: 3.3851 - val_loss: 2.4326
Epoch 4/30
5955/5955 [==============================] - 4s 708us/step - loss: 1.8853 - val_loss: 1.6675
Epoch 5/30
5955/5955 [==============================] - 4s 709us/step - loss: 2.8640 - val_loss: 7.3529
Epoch 6/30
5955/5955 [==============================] - 4s 703us/step - loss: 3.0463 - val_loss: 4.9961
Epoch 7/30
5955/5955 [==============================] - 4s 710us/step - loss: 2.3872 - val_loss: 11.0147
Epoch 8/30
5955/5955 [==============================] - 4s 707us/step - loss: 1.8604 - val_loss: 10.1437
Epoch 9/30
5955/5955 [==============================] - 4s 710us/step - loss: 1.3925 - val_loss: 2.7661
Epoch 10/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.9390 - val_loss: 1.0866
Epoch 11/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.6431 - val_loss: 3.0361
Epoch 12/30
5955/5955 [==============================] - 4s 706us/step - loss: 0.6310 - val_loss: 0.7139
Epoch 13/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.6697 - val_loss: 0.6138
Epoch 14/30
5955/5955 [==============================] - 4s 706us/step - loss: 0.4382 - val_loss: 0.4046
Epoch 15/30
5955/5955 [==============================] - 4s 705us/step - loss: 0.2560 - val_loss: 0.3298
Epoch 16/30
5955/5955 [==============================] - 4s 705us/step - loss: 0.1998 - val_loss: 0.2658
Epoch 17/30
5955/5955 [==============================] - 4s 705us/step - loss: 0.1383 - val_loss: 0.3698
Epoch 18/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.2023 - val_loss: 0.2424
Epoch 19/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.1138 - val_loss: 0.2144
Epoch 20/30
5955/5955 [==============================] - 4s 712us/step - loss: 0.1022 - val_loss: 0.1841
Epoch 21/30
5955/5955 [==============================] - 4s 717us/step - loss: 0.3222 - val_loss: 0.9091
Epoch 22/30
5955/5955 [==============================] - 4s 701us/step - loss: 0.9123 - val_loss: 0.5235
Epoch 23/30
5955/5955 [==============================] - 4s 707us/step - loss: 0.4928 - val_loss: 0.3340
Epoch 24/30
5955/5955 [==============================] - 4s 710us/step - loss: 0.3067 - val_loss: 0.2752
Epoch 25/30
5955/5955 [==============================] - 4s 709us/step - loss: 0.1244 - val_loss: 0.2840
Epoch 26/30
5955/5955 [==============================] - 4s 706us/step - loss: 0.1429 - val_loss: 0.1247
Epoch 27/30
5955/5955 [==============================] - 4s 701us/step - loss: 0.1024 - val_loss: 0.1755
Epoch 28/30
5955/5955 [==============================] - 4s 708us/step - loss: 0.1127 - val_loss: 0.3379
Epoch 29/30
5955/5955 [==============================] - 4s 708us/step - loss: 0.1753 - val_loss: 0.7317
Epoch 30/30
5955/5955 [==============================] - 4s 708us/step - loss: 0.2242 - val_loss: 0.0922'''
