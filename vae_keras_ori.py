import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
from keras.datasets import mnist
from keras import backend as K
import datetime
import ipdb
from keras.models import model_from_json


def sampling(args):
    '''# a keras lambda layer computes arbitrary function on the output of a layer
# so z is effectively combining mean and variance layers through sampling func 
'''
    _mean,_log_var=args
    epsilon=K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
    return _mean+K.exp(_log_var/2)*epsilon


batch_size=100
digit_size=28
original_dim=784
intermediate_dim=256
latent_dim=2

nb_epochs=50
epsilon_std=1.0
datetime = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.datetime.now())
results_dir= "results/vae-keras-ori/".format(batch_size)+ datetime
logdir = results_dir+ "/model"
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print results_dir
######## Encoder
X=Input(batch_shape=(batch_size,original_dim))
h=Dense(intermediate_dim, activation='relu')(X)
z_mean=Dense(latent_dim)(h)
z_log_var=Dense(latent_dim)(h)
z= Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
######## Decoder
h_decoder=Dense(intermediate_dim, activation='relu')
X_bar=Dense(original_dim,activation='sigmoid')

# we instantiate these layers separately so as to reuse them later
h_decoded = h_decoder(z)
X_decoded = X_bar(h_decoded)

def vae_loss(x,x_bar):
    reconst_loss=original_dim*objectives.binary_crossentropy(x, x_bar)
    kl_loss=-0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return reconst_loss + kl_loss

# build and compile model
vae=Model(X , X_decoded)
vae.compile(optimizer='adam', loss=vae_loss)

# load MNIST data for training
(x_train, y_train),(x_test, y_test)=mnist.load_data()

# normalize input and make them float32 to run on GPU
x_train=x_train.astype('float32')/ 255.
x_test=x_test.astype('float32')/255.

# convert 28x28 images into 784-vectors
x_train=x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test=x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# x_train is required for input and loss output as target
vae.fit(x_train,x_train, shuffle=True, epochs=nb_epochs, batch_size=batch_size, validation_data=(x_test, x_test))


# encoder is the inference network
encoder=Model(X, z_mean)

# a 2d plot of 10 digit classes in latent space
x_test_encoded=encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6,6))
plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test)
plt.colorbar()
plt.savefig(results_dir + '/latent_vector_scatter.png', format='png')
plt.close()

# since the generator treats z as an input, we make z an input layer
z_input=Input(shape=(latent_dim,))
_h_decoded=h_decoder(z_input)
_x_decoded=X_bar(_h_decoded)
generator= Model(z_input, _x_decoded)
#####plot sample reconstruction
num_sample = 25
z_sample = np.random.normal(0, 1, (num_sample, latent_dim))
print "z_sample.shape", z_sample.shape
x_decoded = generator.predict(z_sample)      ### start from random
x_test_recon = generator.predict(x_test_encoded[0:num_sample, :])      # use the encoding from test reconstruct
sampled_im = x_decoded.reshape(-1, digit_size, digit_size)
plt.figure()
for ii in range(num_sample):
    ax1 = plt.subplot(5, 5, ii +1)  
    plt.imshow(sampled_im[ii, :, :])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
plt.savefig(results_dir + '/sampled_reconstructon.png', format='png')
plt.close()

#####plot prior
# 2d manifold of images by exploring quantiles of normal dist (using the inverse of cdf)
n=15
figure = np.zeros((digit_size*n, digit_size*n))
grid_x = norm.ppf(np.linspace(0.05,0.95,n))
grid_y = norm.ppf(np.linspace(0.05,0.95,n))
#latent = np.random.normal(0, 1, latent_dim)
for i, yi in enumerate(grid_x):
    for j,xi in enumerate(grid_y):
        latent= np.array([[xi, yi]]) 
        x_decoded=generator.predict(latent)
        digit=x_decoded[0].reshape(digit_size,digit_size)
        figure[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size]=digit
        
plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.savefig(results_dir + '/prior.png', format='png')
plt.close()

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
loaded_model = model_from_json(loaded_model_json, {'batch_size': batch_size, 'latent_dim': latent_dim, 'epsilon_std':epsilon_std})  ## have to give the hyperparams

# load weights into new model
loaded_model.load_weights(logdir+"/model.h5")
print("Loaded model from disk")
 
## evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, x_test, batch_size=batch_size)
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
