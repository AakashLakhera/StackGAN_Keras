# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:08:41 2019

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
gen0_loc = 'Generator0_4.h5'
gen_loc = 'Generator1.h5'

random.seed(time.time())
np.random.seed(int(time.time()+0.5))

_, Emb, captions = load_dataset('birds/test/', 'CUB_200_2011/', 32, True)
print(Emb.shape, len(captions))

gen0 = generator0(Nphi, Ng, Nz)
try:
    gen0.load_weights(gen0_loc)
except:
    print('No Generator0 File Detected! Aborting.')
    exit(0)

gen1 = generator1(Nphi, Ng, Mg, Nres, (64, 64, 3))
try:
    gen1.load_weights(gen_loc)
except:
    print('No Generator1 File Detected! Aborting.')
    exit(0)


lenX = Emb.shape[0]
r = random.randint(0, Emb.shape[1]-1)
phi_t = Emb[:,r,:]
tex = captions[:][r]
eps = np.random.normal(0, 1, [lenX, Ng])
z = np.random.normal(0, 1, [lenX, Nz])
x_img, _ = gen0.predict([phi_t, eps, z])
eps = np.random.normal(0, 1, [lenX, Ng])
x_test, _ = gen1.predict([phi_t, eps, x_img])
for k in range(x_test.shape[0]):
    cv2.imwrite('ResultsII\\'+str(k)+'.jpg', x_test[k])
    f = open('ResultsII\\'+str(k)+'.txt', 'w')
    f.write(tex[k])
    f.close()