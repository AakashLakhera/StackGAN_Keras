# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:28:28 2019

@author: Aakash
"""

import keras
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, UpSampling2D, Flatten, Reshape, Lambda
from keras.layers import BatchNormalization, Activation, LeakyReLU, Input, Add, Multiply, Concatenate
from keras.activations import exponential
from keras.optimizers import Adam
import random
from keras import backend as K
from loadData import *
import time
#import gc

Nphi = 1024
Ng = 128
Nz = 100
Nd = 128
Nx = Ng+Nz
Md = 4
Mg = 16
Nres = 4
batch_size = 64
epochs = 600
start_epoch = 0
learning_rate = 0.0002
gen_loc = 'Generator0.h5'
dis_loc = 'Discriminator0.h5'
random.seed(time.time())
np.random.seed(int(time.time()+0.5))

def generator0(Nphi, Ng, Nz):
    
    Nx = Ng+Nz
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    z = Input(shape=(Nz,))
    
    # Conditioning Augmentation
    musigma = Dense(2*Ng, input_shape=phi_t.shape)(phi_t)
    musigma = LeakyReLU()(musigma)
    mu0 = Lambda(lambda x: x[:,0:Ng])(musigma)
    sigma0 = (Lambda(lambda x: x[:,Ng:])(musigma)) #logsigma
    sigma0 = Activation('exponential')(sigma0)
    tmp = Multiply()([eps, sigma0])
    c0 = Add()([mu0,tmp])
    
    # Making Generator0 Input
    x = Concatenate(axis=1)([c0, z])
    x = Dense(4*Nx, input_shape=x.shape)(x)
    x = Activation('relu')(x)
    x = Reshape([2,2,Nx])(x)
    
    # Code for Generator0 Model- Upsampling
    layerSeq0 = [256, 128, 64, 32, 3]
    in_channels = Nx
    sz = 2
    for i in range(len(layerSeq0)):
        sz = sz<<i
        t = sz<<1
        out_channels = layerSeq0[i]
        x = UpSampling2D(input_shape=(sz,sz,in_channels))(x)
        x = Conv2D(out_channels, 3, padding='same', input_shape=(t,t,in_channels))(x)
        if i < len(layerSeq0)-1:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        in_channels = out_channels
    model = Model(inputs = [phi_t, eps, z], outputs = [x, musigma])
    return model

def generator1(Nphi, Ng, Mg, Nres, imsize):
    
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    x_img = Input(shape=imsize)
    
    # Conditioning Augmentation
    musigma = Dense(2*Ng, input_shape=phi_t.shape)(phi_t)
    musigma = LeakyReLU()(musigma)
    mu1 = Lambda(lambda x: x[:,0:Ng])(musigma)
    sigma1 = (Lambda(lambda x: x[:,Ng:])(musigma)) # logsigma
    sigma1 = Activation('exponential')(sigma1)
    tmp = Multiply()([eps, sigma1])
    c1 = Add()([mu1,tmp])
    
    # Replicating to Mg x Mg x Ng
    c1 = Dense(Ng, input_shape=c1.shape)(c1)
    c1 = Activation('relu')(c1)
    c1 = Reshape([1,1,Ng])(c1)
    c1 = UpSampling2D(input_shape=(1,1,Ng), size=(Mg,Mg))(c1)
    
    # Downsampling
    layerSeq1 = [256, 512]
    in_channels = imsize[2]
    sz = imsize[1]
    img = x_img
    for i in range(len(layerSeq1)):
        out_channels = layerSeq1[i]
        img = Conv2D(out_channels, 4, strides=(2,2), padding='same', input_shape=(sz,sz,in_channels))(img)
        if i!=0:
            img = BatchNormalization()(img)
        img = LeakyReLU()(img)
        sz /= 2
        
    # Concatenate, and feed into residual blocks
    x = Concatenate(axis=3)([c1, img])
    in_channels = out_channels + Ng
    sz = Mg
    for i in range(Nres):
        x_shortcut = x
        x = Conv2D(in_channels, 3, padding='same', input_shape=(sz,sz,in_channels))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(in_channels, 3, padding='same', input_shape=(sz,sz,in_channels))(x)
        x = BatchNormalization()(x)
        x = Add()([x, x_shortcut])
        x = LeakyReLU()(x)
        
    # Code for Generator1 Model- Upsampling
    layerSeq2 = [(512,256), (128, 64), (64, 32), (32,3)] 
    for i in range(len(layerSeq2)):
        sz = sz<<i
        t = sz<<1
        out1 = layerSeq2[i][0]
        x = UpSampling2D(input_shape=(sz,sz,in_channels))(x)
        x = Conv2D(out1, 3, padding='same', input_shape=(t,t,in_channels))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        out_channels = layerSeq2[i][1]
        x = Conv2D(out_channels, 3, padding='same', input_shape=(t,t,out1))(x)
        if i < len(layerSeq2)-1:
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        in_channels = out_channels
    
    model = Model(inputs = [phi_t, eps, x_img], outputs = [x, musigma])
    return model

def discriminator(Nphi, Nd, Md, imsize, layerSeq):
    
    phi_t = Input(shape=(Nphi,))
    img = Input(shape=imsize)
    x1 = Dense(Nd, input_shape=phi_t.shape)(phi_t)
    
    # Replicating to Md x Md x Nd
    x1 = Reshape([1,1,Nd])(x1)
    x1 = UpSampling2D(input_shape=(1,1,Nd), size=(Md,Md))(x1)
    
    # Downsampling
    in_channels = imsize[2]
    sz = imsize[1]
    x0 = img
    for i in range(len(layerSeq)):
        out_channels = layerSeq[i]
        x0 = Conv2D(out_channels, 4, strides=(2,2), padding='same', input_shape=(sz,sz,in_channels))(x0)
        if i!=0:
            x0 = BatchNormalization()(x0)
        x0 = LeakyReLU()(x0)
        sz /= 2
    
    # Concatenate, and feed into 1x1 convolution layer
    x = Concatenate(axis=3)([x0, x1])
    in_channels = out_channels + Nd
    
    x = Conv2D(32, 1, input_shape=(Md, Md, in_channels))(x)
    x = BatchNormalization()(x)
    x = Reshape([Md*Md*32])(x)
    x = Dense(1, input_shape=x.shape, activation='sigmoid')(x)
    model = Model(inputs=[phi_t, img], outputs=[x])
    return model
    
def GAN0(gen, disc, Nphi, Ng, Nz):
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    z = Input(shape=(Nz,))
    disc.trainable = False
    gen_img, musigma = gen([phi_t, eps, z])
    isValid = disc([phi_t, gen_img])
    model= Model(inputs=[phi_t, eps, z], outputs=[isValid, musigma])
    return model

def GAN1(gen, disc, Nphi, Nd, imsize):
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    x_img = Input(shape=imsize)
    disc.trainable = False
    gen_img, musigma = gen([phi_t, eps, x_img])
    isValid = disc([phi_t, gen_img])
    model= Model(inputs=[phi_t, eps, x_img], outputs=[isValid, musigma])
    return model

def KL_loss(y_dummy, musigma):
    mu = musigma[:,:Ng]
    logsigma = musigma[:,Ng:]
    loss = -logsigma + 0.5 * (-1 + K.exp(2*logsigma) + K.square(mu))
    loss = K.mean(loss)
    return loss
    
m = start_epoch//100
learning_rate = 0.0002/(1<<m)
dis_optimizer = Adam(lr=learning_rate, beta_1=0.5)
gen_optimizer = Adam(lr=learning_rate, beta_1=0.5)
gen0 = generator0(Nphi, Ng, Nz)

try:
    gen0.load_weights(gen_loc)
except:
    print('No Generator File Detected!')
dc0 = discriminator(Nphi, Nd, Md, (64,64,3), [64, 128, 256, 512])
try:
    dc0.load_weights(dis_loc)
except:
    print('No Discriminator File Detected!')

dc0.compile(dis_optimizer, loss=['binary_crossentropy'])
gan0 = GAN0(gen0, dc0, Nphi, Ng, Nz)
gan0.compile(gen_optimizer, loss=['binary_crossentropy', KL_loss], loss_weights=[1,1], metrics=None)

'''
gen1 = generator1(Nphi, Nd, Mg, Nres, (64, 64, 3))
dc1 = discriminator(Nphi, Nd, Md, (256,256,3), [32, 64, 128, 256, 512, 512])
gan1 = GAN1(gen1, dc1, Nphi, Md, (64, 64, 3))
'''

X_real, Emb = load_dataset('birds/train/', 'CUB_200_2011/', 64)
print('Embeddings:', Emb.shape,'CUB Dataset:', X_real.shape)

lenX = X_real.shape[0]
iterations = 1+(lenX//batch_size)

print('Starting to Train')
print('The Learning Rate Now is:', K.get_value(dc0.optimizer.lr))
for i in range(start_epoch, epochs):
    rand_shuffle = np.arange(lenX)
    np.random.shuffle(rand_shuffle)
    X_real = X_real[rand_shuffle]
    Emb = Emb[rand_shuffle]
    d_loss = 0
    g_loss = [0, 0, 0]
    for j in range(iterations):
        i1 = j*batch_size
        i2 = i1 + batch_size
        x_real = X_real[i1:i2,:,:]
        phi_t = Emb[i1:i2,random.randint(0, Emb.shape[1]-1),:]
        
        curr_size = int(tuple(x_real.shape)[0])
        d_size = curr_size<<1
        musigma_dummy = np.ones((curr_size,2*Ng))
        real_labels = np.ones((curr_size,1))
        false_labels = np.zeros((curr_size,1))
        
        # Now, get some wrong images
        i3 = (i2 + batch_size)
        if i2 >= lenX or i3 >lenX:
            i2 = 0
            i3 = curr_size    
        x_wrong = X_real[i2:i3,:,:]
        wrong_labels = np.zeros((curr_size,1))
        
        eps = np.random.normal(0, 1, [curr_size, Ng])
        z = np.random.normal(0, 1, [curr_size, Nz])
        x_false, musigma = gen0.predict([phi_t, eps, z])
        
        X = np.concatenate([x_real, x_wrong, x_false, ], axis = 0)
        Phi_t = np.concatenate([phi_t, phi_t, phi_t], axis=0)
        Labels = np.concatenate([real_labels, wrong_labels, false_labels], axis=0)
        
        shuff = np.arange(3*curr_size)
        np.random.shuffle(shuff)
        X = X[shuff]
        Phi_t = Phi_t[shuff]
        Labels = Labels[shuff]
        
        d_loss += dc0.train_on_batch([Phi_t[0:curr_size], X[0:curr_size]], [Labels[0:curr_size]])
        d_loss += dc0.train_on_batch([Phi_t[curr_size:d_size], X[curr_size:d_size]], [Labels[curr_size:d_size]])
        d_loss += dc0.train_on_batch([Phi_t[d_size:], X[d_size:]], [Labels[d_size:]])
        
        _, musigma =  gan0.predict([phi_t, eps, z])
        loss = gan0.train_on_batch([phi_t, eps, z], [real_labels, musigma_dummy])
        for k in range(len(loss)):
            g_loss[k] += loss[k]
        
        
        if ((j+1) % 30 == 0) or ((j+1) == iterations):
            print((i+1), (j+1), d_loss, g_loss)
            d_loss = 0
            g_loss = [0, 0, 0]
            #print('Memory Recollected-',gc.collect())
    
    gen0.save_weights(gen_loc)
    dc0.save_weights(dis_loc)
    if (i+1)%100 == 0:
        learning_rate /= 2
        K.set_value(dc0.optimizer.lr, learning_rate)
        K.set_value(gan0.optimizer.lr, learning_rate)
        print('The Learning Rate Now is:', K.get_value(dc0.optimizer.lr))
        

'''
x_real = K.constant(np.random.randint(0, 255, (128,64,64,3)), dtype='float')
phi_t = K.constant(np.random.rand(128,Nphi))

#x_real = np.random.randint(0, 255, (128,64,64,3))
#x_real = x_real.astype('float')
#phi_t = np.random.rand(128, Nphi)

batch_size = int(tuple(x_real.shape)[0])
epochs = 10

for i in range(epochs):
    batch_size = int(tuple(x_real.shape)[0])
    d_batch_size = batch_size << 1
    musigma_dummy = K.ones((batch_size,2*Ng))
    real_labels = K.ones((batch_size,1))
    false_labels = K.zeros((batch_size,1))
    eps = K.constant(np.random.normal(0, 1, [batch_size, Ng]))
    z = K.constant(np.random.normal(0, 1, [batch_size, Nz]))
    x_false, musigma = gen0([phi_t, eps, z])
    #print(K.get_value(K.mean(musigma, axis=1)))
    #eps = np.random.multivariate_normal(np.zeros(Ng), np.identity(Ng), batch_size)
    #z = np.random.multivariate_normal(np.zeros(Nz), np.identity(Nz), batch_size)
    #x_false, musigma = gen0.predict([phi_t, eps, z])
    X = K.concatenate([x_real, x_false], axis = 0)
    Phi_t = K.concatenate([phi_t, phi_t], axis=0)
    loss = dc0.train_on_batch([Phi_t, X], [K.concatenate([real_labels, false_labels], axis=0)])
    print((i+1), loss)
    #X = np.concatenate([x_real, x_false], axis = 0)
    #Phi_t = np.concatenate([phi_t, phi_t], axis=0)
    #dc0.fit([Phi_t, X], [np.concatenate([np.ones((batch_size,1)), np.zeros((batch_size,1))])], d_batch_size)
    _, musigma =  gan0([phi_t, eps, z])
    loss = gan0.train_on_batch([phi_t, eps, z], [real_labels, musigma_dummy])
    print((i+1), loss)
    #eps = K.constant(np.random.multivariate_normal(np.zeros(Ng), np.identity(Ng), 20))
    #z = K.constant(np.random.multivariate_normal(np.zeros(Nz), np.identity(Nz), 20))
    #loss = gan0.fit([Phi_t[:batch_size,:], eps, z], [np.ones((batch_size, 1)), np.ones((batch_size, 2*Ng))], batch_size)
    
#y = dc0([Phi_t[:batch_size,:], x_real])
#print(K.get_value(y))
'''



