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

nb_epochs = 2
epsilon_std = 1.0
data_dir = "data/npy_data/ds_8testF751testN1501_average.npz"

datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir= "results/vae-keras-ori-eeg/".format(batch_size)+ datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
################## Encoder ##################
X = Input(batch_shape = (batch_size, original_dim*channel))
print "X", X
net  =  Dense(intermediate_dim, activation = 'relu')(X)
print "enc", net
z_mean = Dense(latent_dim)(net)
z_log_var = Dense(latent_dim)(net)
z = Lambda(sampling, output_shape = (latent_dim,))([z_mean, z_log_var])

#################### Decoder graph #########################
h_decoder = Dense(intermediate_dim, activation = 'relu')
X_bar = Dense(original_dim * channel, activation = 'sigmoid')
# we instantiate these layers separately so as to reuse them later
dec_net = h_decoder(z)
print "dec_h", dec_net 
X_reconstruction  = X_bar(dec_net )
print "X_decoded", X_reconstruction

############ loss ######################3
def vae_loss(x,x_rec):
    reconst_loss = original_dim*objectives.binary_crossentropy(x, x_rec)
    kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
    return reconst_loss - kl_loss
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
vae.fit(x_train,x_train, shuffle = True, epochs = nb_epochs, batch_size = batch_size, validation_data = (x_test, x_test))

# encoder is the inference network
encoder = Model(X, z_mean)

ipdb.set_trace()
# a 2d plot of 10 digit classes in latent space
x_test_encoded = encoder.predict(x_test, batch_size = batch_size)
plt.figure(figsize = (6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c = y_test)
plt.colorbar()


# since the generator treats z as an input, we make z an input layer
z_input = Input(shape = (latent_dim,))
_h_decoded = h_decoder(z_input)
_x_decoded = X_bar(_h_decoded)
generator =  Model(z_input, _x_decoded)

#####plot sample reconstruction
z_sample = np.array([np.random.normal(0, 1, latent_dim)])
print z_sample
x_decoded = generator.predict(z_sample)
sampled_im = x_decoded[0].reshape(original_dim, channel )
plt.figure()
plt.imshow(sampled_im, cmap = 'Greys_r')


#####plot prior
# 2d manifold of images by exploring quantiles of normal dist (using the inverse of cdf)
n = 15
figure  =  np.zeros((2*n, original_dim))

grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))
#latent = np.random.normal(0, 1, latent_dim)
for i, yi in enumerate(grid_x):
    for j,xi in enumerate(grid_y):
        latent =  np.array([[xi, yi]]) 
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(original_dim, )
        figure[i*digit_size:(i+1)*digit_size, 
              j*digit_size:(j+1)*digit_size] = digit
        
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
Epoch 1/50
60000/60000 [==============================] - 8s 130us/step - loss: 201.0283 - val_loss: 174.7658
Epoch 15/50
60000/60000 [==============================] - 8s 128us/step - loss: 153.9201 - val_loss: 154.6933
Epoch 25/50
60000/60000 [==============================] - 8s 129us/step - loss: 150.8378 - val_loss: 152.4690
Epoch 40/50
60000/60000 [==============================] - 8s 129us/step - loss: 148.3840 - val_loss: 151.0905
Epoch 45/50
60000/60000 [==============================] - 8s 134us/step - loss: 147.8082 - val_loss: 150.8328
Epoch 50/50
60000/60000 [==============================] - 8s 126us/step - loss: 147.3580 - val_loss: 150.6444'''
