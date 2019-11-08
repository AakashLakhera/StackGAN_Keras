# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:27:17 2019

@author: Joydeep Nag
"""

import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

def inception_score(yval, splits = 1,eps = 1E-16):
    scores = []
    for i in range(splits):
        istart = i * yval.shape[0] // splits
        iend = (i + 1) * yval.shape[0] // splits
        x = yval[istart:iend, :]
        kl = x *(np.log(x+eps) - np.log(np.expand_dims(np.mean(x+eps, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


images = np.ones((50, 256, 256, 3))

model = InceptionV3(include_top=False, input_shape=(256,256,3))
images = preprocess_input(images.astype('float32'))
y = model.predict(images)
print(inception_score(y))