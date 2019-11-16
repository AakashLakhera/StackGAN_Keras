# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:28:28 2019

@author: Aakash
"""

import random
import numpy as np
import time
from loadData import *
from modelsGAN import *
import cv2

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
gen_loc = 'Generator0_4.h5'
dis_loc = 'Discriminator0_4.h5'
random.seed(time.time())
np.random.seed(int(time.time()+0.5))

def KL_loss(y_dummy, musigma):
    global Ng
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

print('Loading Dataset...')
X_real, Emb, _ = load_dataset('birds/train/', 'CUB_200_2011/', 64)
X_real = (X_real-127.5)/127.5

testEmb, c_, d_ = extract_aux_info('birds/test/')
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
        real_labels = np.ones((curr_size,1))*0.9
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
        
        X = np.concatenate([x_real, x_false, x_wrong], axis = 0)
        Phi_t = np.concatenate([phi_t, phi_t, phi_t], axis=0)
        Labels = np.concatenate([real_labels, false_labels, wrong_labels], axis=0)
        
        #shuff = np.arange(3*curr_size)
        #np.random.shuffle(shuff)
        #X = X[shuff]
        #Phi_t = Phi_t[shuff]
        #Labels = Labels[shuff]
        
        d_loss += dc0.train_on_batch([Phi_t[0:curr_size], X[0:curr_size]], [Labels[0:curr_size]])
        d_loss += dc0.train_on_batch([Phi_t[curr_size:d_size], X[curr_size:d_size]], [Labels[curr_size:d_size]])
        d_loss += dc0.train_on_batch([Phi_t[d_size:], X[d_size:]], [Labels[d_size:]])
        
        loss = gan0.train_on_batch([phi_t, eps, z], [real_labels, musigma_dummy])
        for k in range(len(loss)):
            g_loss[k] += loss[k]
        
        
        if ((j+1) % 30 == 0) or ((j+1) == iterations):
            d_loss /= 30
            for k in range(len(g_loss)):
                g_loss[k] /= 30 
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
        emb = testEmb[:,random.randint(0, testEmb.shape[1]-1),:]
        eps = np.random.normal(0, 1, [testEmb.shape[0], Ng])
        z = np.random.normal(0, 1, [testEmb.shape[0], Nz])
        x_test, _ = gen0.predict([emb, eps, z])
        x_test = 127.5*(x_test + 1)
        for k in range(x_test.shape[0]):
            cv2.imwrite('ResultsI\\'+str(k)+'.jpg', x_test[k])
        
