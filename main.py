# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 08:48:10 2019
@author: Cody, Sonal and Saikat
"""

# For timestamp
import os
import random
import time

#Image related library
import numpy as np
import skimage.io
import cv2
from PIL import Image

# Tensorflow library 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# These library used for reading file and folder
from os.path import isfile,join
from os import listdir
from keras import backend as K


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_MAX_VAL = 255
IMG_THRESHOLD = 0.4
# Updated to work  with tensorflow 2.0

# Data class, all the functionality like fetching data 
# and data preprocessing done inside the class
class Data():
	def __init__(self, data_dir):
		images_list =[]
		labels_list = []
		# Get Image and label folder path
		label_dir = os.path.join(data_dir, "Labels")
		image_dir = os.path.join(data_dir, "Images")
		self.image_size = 128
		examples = 0
		print("Loading images.....")
		# Get the file name of image and label
		onlyImagefiles = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
		onlyLabelfiles = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
		#Sort them so that it can be used for mapping later on.
		onlyImagefiles.sort()
		onlyLabelfiles.sort()
		# Since number of images and label are same, use the number of images to iterate
		for i in range (len(onlyImagefiles)):
			#Concatenate the folder and file firectory to get the full directory path
			image = cv2.imread(os.path.join(image_dir,onlyImagefiles[i]))
			label = cv2.imread(os.path.join(label_dir,onlyLabelfiles[i]),cv2.IMREAD_GRAYSCALE)
			#image= cv2.resize(image, (self.image_size, self.image_size))
			#label= cv2.resize(label, (self.image_size, self.image_size))
			
			# Hack alert: This is a hardcoded number, purpose of this number is to only fetch
			# region of interest in the image. 
			image = image[96:224,96:224,:]
			label = label[96:224,96:224]
			
			# Preprocessing of the image
			# Label is true for the region greater than threshold
			label = label>IMG_THRESHOLD*100

			#Regularize the image between 0-1
			image = image/IMG_MAX_VAL
			#Remove the extra dimension and change the type to int
			label = label[...,None]
			label = label.astype(np.int32)

			images_list.append(image)
			labels_list.append(label)
			examples = examples +1
							
		print("finished loading images")
		self.examples = examples
		print("Number of examples found: ", examples)
		self.images = np.array(images_list)
		self.labels = np.array(labels_list)


# Base Directory Directory 
base_dir= 'Data'

# Training on first domain
train_dir = os.path.join(base_dir,'Train')
# Testing on second domain
test_dir = os.path.join(base_dir,'Test')
# Testing on first domain
real_dir = os.path.join(base_dir,'Real')

image_size = 128

# Fetch and process the image
def PreProcessImages():
	train_data = Data(train_dir)
	test_data = Data(test_dir)
	real_data = Data(real_dir)

	return train_data,  test_data, real_data

# Used f1 metric formula to calculate
def f1_metric(y_true, y_pred):
	y_true = y_true >IMG_THRESHOLD
	y_pred = y_pred>IMG_THRESHOLD
	y_true = tf.dtypes.cast(y_true,tf.float32)
	y_pred = tf.dtypes.cast(y_pred,tf.float32)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
	return f1_val


def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
	p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
	return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
	us = keras.layers.UpSampling2D((2, 2))(x)
	concat = keras.layers.Concatenate()([us, skip])
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
	return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
	c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
	return c
	
def UNet():
	f = [16, 32, 64, 128, 256]
	inputs = keras.layers.Input((image_size, image_size, 3))
	
	p0 = inputs
	c1, p1 = down_block(p0, f[0]) #128 -> 64
	c2, p2 = down_block(p1, f[1]) #64 -> 32
	c3, p3 = down_block(p2, f[2]) #32 -> 16
	c4, p4 = down_block(p3, f[3]) #16->8
	
	bn = bottleneck(p4, f[4])
	
	u1 = up_block(bn, c4, f[3]) #8 -> 16
	u2 = up_block(u1, c3, f[2]) #16 -> 32
	u3 = up_block(u2, c2, f[1]) #32 -> 64
	u4 = up_block(u3, c1, f[0]) #64 -> 128
	
	outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
	model = keras.models.Model(inputs, outputs)
	return model

class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		#show_predictions()
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))\

def dice_coef(y_true, y_pred, smooth=1):
	y_true = y_true >IMG_THRESHOLD
	y_pred = y_pred>IMG_THRESHOLD
	y_true = tf.dtypes.cast(y_true,tf.float32)
	y_pred = tf.dtypes.cast(y_pred,tf.float32)

	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def TrainUnet():
#############################Start Unet training###############################
	epochs = 20
	batch_size = 1
	model = UNet()
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[f1_metric,dice_coef])
	model.summary()
	
	train_data,  test_data, real_data= PreProcessImages()
	
	if not os.path.exists("UNetW.h5"):
		model_history = model.fit(train_data.images,train_data.labels,validation_split=0.3, epochs=epochs)
		loss = model_history.history['loss']
		val_loss = model_history.history['val_loss']
		dice_Val= model_history.history['dice_coef']
		F1_Val= model_history.history['f1_metric']
	else:
		model.load_weights("UNetW.h5")
	
	resultCross = model.predict(test_data.images)
	resultSame = model.predict(real_data.images)


	resultCross = resultCross >IMG_THRESHOLD
	resultSame = resultSame >IMG_THRESHOLD

	score = model.evaluate(test_data.images,test_data.labels)

	print("Cross Domain Loss: "+str(score[0]))
	print("Cross Domain F1 score: "+str(score[1]))
	print("Cross Domain Dice Coef: "+str(score[2]))

	score = model.evaluate(real_data.images,real_data.labels)

	print("Real Domain Loss: "+str(score[0]))
	print("Real Domain F1 score: "+str(score[1]))
	print("Real Domain Dice Coef: "+str(score[2]))

	for i in range (resultSame.shape[0]):
		img = resultSame[i]
		img = img>IMG_THRESHOLD
		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/SamePredicted_"+str(i)+".jpg",img)
		img = real_data.labels[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/SameGround_"+str(i)+".jpg",img)
		img = real_data.images[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/SameImage_"+str(i)+".jpg",img)
		img = resultCross[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/CrossPredicted_"+str(i)+".jpg",img)
		img = test_data.labels[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/CrossGround_"+str(i)+".jpg",img)
		img = test_data.images[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("Results/Images/CrossImage_"+str(i)+".jpg",img)

	

	epochs = range(epochs)
	if not os.path.exists("UNetW.h5"):
		
		fig1= plt.figure()
		plt.plot(epochs, loss, 'r', label='Training loss')
		plt.plot(epochs, val_loss, 'bo', label='Validation loss')
		plt.title('Training and Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss Value')
		plt.ylim([0, 1])
		plt.legend()
		plt.show()
		
		fig2= plt.figure()
		plt.plot(epochs, dice_Val, 'r', label='Dice Coefficient ')
		plt.plot(epochs, F1_Val, 'b', label='f1 Metric')
		plt.title('Dice Coefficient and F1 Metrics')
		plt.xlabel('Epoch')
		plt.ylabel('Metrics Value')
		plt.ylim([0, 1])
		plt.legend()
		plt.show()

	model.save_weights("UNetW.h5")
	return model
		
def main():
	model = TrainUnet()

if __name__ == '__main__':
	main()

