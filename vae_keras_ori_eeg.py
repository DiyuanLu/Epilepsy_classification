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
from tensorflow.python import debug as tfdb          ## for debuging
from keras.models import model_from_json
import functions as func

def sampling(args):
    '''# a keras lambda layer computes arbitrary function on the output of a layer
# so z is effectively combining mean and variance layers through sampling func 
'''
    _mean,_log_var = args
    epsilon = K.random_normal(shape = (batch_size, latent_dim), mean = 0., stddev = epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon


batch_size =  15            # train: 5955    test:  1485 samples
ifaverage = True                    ###False   ###
if ifaverage:
    channel = 1
else:
    channel  = 2
original_dim = 1280
seq_shape = (original_dim, channel)
intermediate_dim = 256
latent_dim = 2

nb_epochs = 30
epsilon_std = 1.0
data_dir = "data/npy_data/testF751testN1501_aver_zscore.npz"

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir= "results/vae-keras-ori-eeg/".format(batch_size)+ datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print results_dir

#sess = K.get_session()
#sess = tfdb.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)
################## Encoder ##################
X = Input(batch_shape = (batch_size, original_dim*channel))
net  =  Dense(intermediate_dim, activation = 'relu')(X)
z_mean = Dense(latent_dim)(net)
z_log_var = Dense(latent_dim)(net)
z = Lambda(sampling, output_shape = (latent_dim,))([z_mean, z_log_var])

#################### Decoder graph #########################
h_decoder = Dense(intermediate_dim, activation = 'relu')
X_bar = Dense(original_dim * channel, activation = 'sigmoid')
# we instantiate these layers separately so as to reuse them later
dec_net = h_decoder(z)
X_reconstruction  = X_bar(dec_net )

############ loss ######################3
def vae_loss(x, x_rec):
    reconst_loss = objectives.binary_crossentropy(x, x_rec)
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    return reconst_loss + kl_loss
# build and compile model
vae = Model(X , X_reconstruction)
vae.compile(optimizer = 'adam', loss = vae_loss)

# load data for training
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
data_file = np.load(data_dir)
x_train, y_train, x_test, y_test = data_file["x_train"], data_file["y_train"], data_file["x_test"], data_file["y_test"]

# normalize input and make them float32 to run on GPU
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# convert 28x28 images into 784-vectors
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# x_train is required for input and loss output as target
vae.fit(x_train, x_train, shuffle = True, epochs = nb_epochs, batch_size = batch_size, validation_data = (x_test, x_test))

############### encoder is the inference network
encoder = Model(X, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size = batch_size)
x_decoded = generator.predict(z_sample)
plt.figure(figsize = (6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c = np.argmax(y_test, axis=1))
plt.colorbar()
plt.savefig(results_dir + '/latent_scatter.png', format='png')
plt.close()


# since the generator treats z as an input, we make z an input layer
z_input = Input(shape = (latent_dim,))
_h_decoded = h_decoder(z_input)
_x_decoded = X_bar(_h_decoded)
generator =  Model(z_input, _x_decoded)

#####plot sample reconstruction
z_sample = np.array([np.random.normal(0, 1, latent_dim)])
print z_sample
x_decoded = generator.predict(z_sample)      ### start from random
x_test_recon = generator.predict(x_test_encoded[0:5, :])      # use the encoding from test reconstruct
sampled_im = x_decoded[0].reshape(original_dim, channel )
plt.figure()
plt.plot(sampled_im, 'c', label='rand_latent_recon')
plt.legend()
plt.savefig(results_dir + '/sampled_reconstructon.png', format='png')
plt.close()

ipdb.set_trace()
plt.figure()
for ii in range(5)
    ax1 = plt.subplot(5,1, ii +1)  
    plt.plot(x_test_recon[ii:], label='test_recon_{}'.format(ii))
    plt.plot(x_test[ii:], label='test_original'.format(ii))
plt.legend()
plt.savefig(results_dir + '/test_reconstructons.png', format='png')
plt.close()

#####plot prior
# 2d manifold of images by exploring quantiles of normal dist (using the inverse of cdf)
n = 3
figure  =  plt.subplot()
grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))
#latent = np.random.normal(0, 1, latent_dim)
gs = plt.GridSpec(n, n, hspace=0)
fig = plt.figure()
other_axes = [fig.add_subplot(gs_top[i,:], sharex=ax) for i in range(1: )]
other_axes[0].plot(data12, 'c', label="non-focal")


for i, yi in enumerate(grid_x):
    for j,xi in enumerate(grid_y):
        latent =  np.array([[xi, yi]]) 
        x_decoded = generator.predict(z_sample)
        if i == 0 and j == 0:
            ax = fig.add_subplot(gs[0, :])
            ax.plot(x_decoded)
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            other_axes[i + j].plot(x_decoded)
            
        
plt.figure(figsize = (10,10))
plt.imshow(figure, cmap = 'Greys_r')
plt.show()
ipdb.set_trace()

# serialize model to JSON
model_json = vae.to_json()
with open(logdir + "/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
vae.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
## load json and create model
json_file = open(logdir+'/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(logdir+"/model.h5")
print("Loaded model from disk")
 
## evaluate loaded model on test data
loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
score = loaded_model.evaluate(X, Y, verbose = 0)
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
