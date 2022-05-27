import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import PIL
import PIL.Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator  
from keras.models import Model
from keras.layers import Dense

from sklearn.model_selection import KFold

from classification_models.tfkeras import Classifiers

def preprocess(images, labels):
    return preprocess_input(images), labels

def expand_d(images, labels):
     return tf.expand_dims(images, axis=0), tf.expand_dims(labels, axis=0)

hyperparam = {}
hyperparam["PixelRangeShear"] = 5;         # max. xy translation (in pixels) for image augmenter

train_img_dir = 'E:/split_dataset/train'
test_img_dir = 'E:/split_dataset/test'

img_height = 224
img_width = 224

train_images = tf.keras.utils.image_dataset_from_directory(
  train_img_dir,
  labels='inferred',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=64,
  label_mode='categorical'
  )

test_images = tf.keras.utils.image_dataset_from_directory(
  test_img_dir,
  labels='inferred',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=64,
  label_mode='categorical'
  )

hw_factor = hyperparam["PixelRangeShear"]/224
flip_layer = layers.RandomFlip("horizontal_and_vertical")
translation_layer = layers.RandomTranslation(height_factor=hw_factor, width_factor=hw_factor)
resizing_layer = layers.Resizing(img_height, img_width)

train_images_r = train_images.map(lambda x, y: (resizing_layer(x), y))
train_images_prep = train_images_r.map(preprocess)
#train_images_prep_exp = train_images_prep.map(expand_d)
train_images_prep_exp = train_images_prep

test_images_r = train_images.map(lambda x, y: (resizing_layer(x), y))
test_images_prep = test_images_r.map(preprocess)
#test_images_prep_exp = test_images_prep.map(expand_d)
test_images_prep_exp = test_images_prep

class_names = train_images.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_set_cache = train_images_prep_exp.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_set_cache = test_images_prep_exp.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

train_set_cache_da = train_set_cache.map(lambda x, y: (flip_layer(x), y))
train_set_cache_da = train_set_cache_da.map(lambda x, y: (translation_layer(x), y))

ResNet18, preprocess_input = Classifiers.get('resnet18')
model_orig = ResNet18((224, 224, 3), weights='imagenet')
new_layer_output = Dense(len(class_names), activation='softmax', name='predictions')
model = Model(model_orig.input, new_layer_output(model_orig.layers[-3].output))

model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
ep=5
model.fit(train_set_cache_da, validation_data=test_set_cache, epochs=ep, validation_freq=1)
