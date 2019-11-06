# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:28:28 2019

@author: Aakash
"""

import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, UpSampling2D, Flatten, Reshape, Lambda 
from keras.layers import BatchNormalization, Activation, LeakyReLU, Input, Add, Multiply, Concatenate
from keras.optimizers import Adam
from keras import backend as K

Nphi = 1000
Ng = 128
Nz = 100
Nd = 128
Nx = Ng+Nz
Md = 4
Mg = 16
Nres = 4

def generator0(Nphi, Ng, Nz):
    
    Nx = Ng+Nz
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    z = Input(shape=(Nz,))
    
    # Conditioning Augmentation
    musigma = Dense(2*Ng, input_shape=phi_t.shape)(phi_t)
    musigma = LeakyReLU()(musigma)
    mu0 = Lambda(lambda x: x[:,0:Ng])(musigma)
    sigma0 = Lambda(lambda x: x[:,Ng:])(musigma)
    tmp = Multiply()([eps, sigma0])
    c0 = Add()([mu0,tmp])
    
    # Making Generator0 Input
    x = Concatenate(axis=1)([c0, z])
    x = Reshape([1,1,Nx])(x)
    
    # Code for Generator0 Model- Upsampling
    layerSeq0 = [256, 128, 64, 32, 16, 3]
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
    sigma1 = Lambda(lambda x: x[:,Ng:])(musigma)
    tmp = Multiply()([eps, sigma1])
    c1 = Add()([mu1,tmp])
    
    # Replicating to Mg x Mg x Ng
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
    gen_img, musigma = gen([phi_t, eps, z])
    isValid = disc([phi_t, gen_img])
    model= Model(inputs=[phi_t, eps, z], outputs=[isValid, musigma])
    return model

def GAN1(gen, disc, Nphi, Nd, imsize):
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    x_img = Input(shape=imsize)
    gen_img, musigma = gen([phi_t, eps, x_img])
    isValid = disc([phi_t, gen_img])
    model= Model(inputs=[phi_t, eps, x_img], outputs=[isValid, musigma])
    return model

def KL_loss(musigma, y_dummy):
    mu = musigma[:,:Ng]
    sigma = musigma[:,Ng:]
    loss = -K.log(sigma) + 0.5 * (-1 + K.square(sigma) + K.square(mu))
    loss = K.mean(loss)
    return loss
    

dis_optimizer = Adam(lr=0.1, beta_1=0.5, beta_2=0.999)
gen_optimizer = Adam(lr=0.1, beta_1=0.5, beta_2=0.999)
gen0 = generator0(Nphi, Ng, Nz)
gen0.compile(gen_optimizer, loss='binary_crossentropy')
dc0 = discriminator(Nphi, Nd, Md, (64,64,3), [64, 128, 256, 512])
dc0.compile(dis_optimizer, loss='binary_crossentropy')
gan0 = GAN0(gen0, dc0, Nphi, Ng, Nz)
gan0.compile(gen_optimizer, loss=['binary_crossentropy', KL_loss], loss_weights=[1, 1])

'''
gen1 = generator1(Nphi, Nd, Mg, Nres, (64, 64, 3))
dc1 = discriminator(Nphi, Nd, Md, (256,256,3), [32, 64, 128, 256, 512, 512])
gan1 = GAN1(gen1, dc1, Nphi, Md, (64, 64, 3))
'''
x_real = K.constant(np.random.randint(0, 255, (10,64,64,3)), dtype='float')
phi_t = K.constant(np.random.rand(1,Nphi))

epochs = 10
for i in range(epochs):
    eps = K.constant(np.random.multivariate_normal(np.zeros(Ng), np.identity(Ng), 10))
    z = K.constant(np.random.multivariate_normal(np.zeros(Nz), np.identity(Nz), 10))
    gen0.trainable = False
    dc0.trainable = True
    x_false, musigma = gen0([phi_t, eps, z])
    X = K.concatenate([x_real, x_false], axis = 0)
    Phi_t = K.concatenate([phi_t for i in range(20)], axis=0)
    loss = dc0.train_on_batch([Phi_t, X], K.constant([1 for x in range(10)] + [0 for x in range(10)]))
    print((i+1), loss)
    gen0.trainable = True
    dc0.trainable = False
    #eps = K.constant(np.random.multivariate_normal(np.zeros(Ng), np.identity(Ng), 20))
    #z = K.constant(np.random.multivariate_normal(np.zeros(Nz), np.identity(Nz), 20))
    loss = gan0.train_on_batch([Phi_t[:10,:], eps, z], [K.constant([1 for x in range(10)]), K.ones((10, 2*Ng))])
    print((i+1), loss)
    
y = dc0([Phi_t[:10,:], x_real])
print(K.get_value(y))





