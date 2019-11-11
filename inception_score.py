# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:27:17 2019

@author: Joydeep Nag
"""

import numpy as np
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

#Create the model
classes = 256

base_model = InceptionV3(include_top=False, input_shape=(256,256,3))
x = base_model.output
x = GlobalAveragePooling2D(name = 'avg_pool')(x)
x = Dropout(0.3)(x)
predictions = Dense(classes,activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)

for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer = 'Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])

#Training
train_data = ImageDataGenerator(preprocessing_function=preprocess_input,
    rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
    shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

train_gen = train_data.flow_from_directory('./train/',
    target_size = (256,256),color_mode='rgb',batch_size=32,
    class_mode='categorical',shuffle=True)

step_size = train_gen.n//train_gen.batch_size
model.fit_generator(generator=train_gen,steps_per_epoch=step_size,
     epochs=10)

#Find the Inception score
images = np.ones((50, 256, 256, 3))
images = preprocess_input(images.astype('float32'))
y = model.predict(images)
print(inception_score(y))
