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
    model = Model(inputs = [phi_t, eps, z], outputs = [x])
    return model

def generator1(Nphi, Ng, Mg, Nres, imsize):
    
    phi_t = Input(shape=(Nphi,))
    eps = Input(shape=(Ng,))
    x_img = Input(shape=imsize)
    
    # Conditioning Augmentation
    musigma = Dense(2*Ng, input_shape=phi_t.shape)(phi_t)
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
    
    model = Model(inputs = [phi_t, eps, x_img], outputs = [x])
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
    
    
eps = K.constant(np.random.multivariate_normal(np.zeros(Ng), np.identity(Ng), 1))
z = K.constant(np.random.multivariate_normal(np.zeros(Nz), np.identity(Nz), 1))
phi_t = K.constant(np.random.rand(1,1000))

gen0 = generator0(Nphi, Ng, Nz)
dc0 = discriminator(Nphi, Nd, Md, (64,64,3), [64, 128, 256, 512])
y = gen0([phi_t, eps, z])
yd = dc0([phi_t, y])
print(y.shape, yd.shape)
'''
gen0 = generator0(Nphi, Ng, Nz)
gen0.summary()
gen1 = generator1(Nphi, Ng, Mg, Nres, (64,64,3))
gen1.summary()
y = gen0([phi_t, eps, z])
y = gen1([phi_t, eps, y])
print(y.shape)
'''
'''
# Code for Discriminator Model
# First we Downsample
downSample0 = Sequential()
layerSeq0 = [16, 32, 64, 128]
in_channels = 3
sz = 64
for i in range(len(layerSeq0)):
    out_channels = layerSeq0[i]
    downSample0.add(Conv2D(out_channels, 4, strides=(2,2), input_shape=(sz,sz,in_channels)))
    if i!=0:
        downSample0.add(BatchNormalization())
    downSample0.add(LeakyReLU())
    sz /= 2
downSample0.build(input_shape=(1,1,Nx))
downSample0.summary()
'''
# Next, we concatenate with the text embedding



