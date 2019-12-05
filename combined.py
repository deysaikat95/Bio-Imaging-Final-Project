import os
import random
import time
import numpy as np

import skimage.io
import cv2
from PIL import Image
import torch

import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt

from os.path import isfile,join
from os import listdir
from keras import backend as K


AUTOTUNE = tf.data.experimental.AUTOTUNE
# Updated to work  with tensorflow 2.0
class GetData():
	def __init__(self, data_dir):	
		images_list =[]		
		labels_list = []		
		self.source_list = []
		label_dir = os.path.join(data_dir, "Labels")
		image_dir = os.path.join(data_dir, "Images")
		self.image_size = 128
		examples = 0
		print("loading images")
		onlyImagefiles = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
		onlyLabelfiles = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
		onlyImagefiles.sort()
		onlyLabelfiles.sort()

		for i in range (len(onlyImagefiles)):
			image = cv2.imread(os.path.join(image_dir,onlyImagefiles[i]))
			#im = Image.open(os.path.join(label_dir,onlyLabelfiles[i]),cv2.IMREAD_GRAYSCALE)
			#label = np.array(im)
			label = cv2.imread(os.path.join(label_dir,onlyLabelfiles[i]),cv2.IMREAD_GRAYSCALE)
			#image= cv2.resize(image, (self.image_size, self.image_size))
			#label= cv2.resize(label, (self.image_size, self.image_size))
			image = image[96:224,96:224,:]
			label = label[96:224,96:224]
			#cv2.imwrite("Pre_"+str(i)+".jpg",label)
			#image = image[...,0][...,None]/255
			label = label>40
			image = image/255
			#image = image[...,None]
			label = label[...,None]
			label = label.astype(np.int32)
			#label = label*255
			#cv2.imwrite("Post_"+str(i)+".jpg",label)
			images_list.append(image)
			labels_list.append(label)
			examples = examples +1
							
		print("finished loading images")
		self.examples = examples
		print("Number of examples found: ", examples)
		self.images = np.array(images_list)
		self.labels = np.array(labels_list)

	def next_batch(self, batch_size):
	
		if len(self.source_list) < batch_size:
			new_source = list(range(self.examples))
			random.shuffle(new_source)
			self.source_list.extend(new_source)

		examples_idx = self.source_list[:batch_size]
		del self.source_list[:batch_size]

		return self.images[examples_idx,...], self.labels[examples_idx,...]


# Base Directory Directory 
base_dir= 'Data'

# Training and Test Directories 
train_dir = os.path.join(base_dir,'Train')
test_dir = os.path.join(base_dir,'Test')
real_dir = os.path.join(base_dir,'Real')

BATCH_SIZE = 1
BUFFER_SIZE = 1000
image_size = 128
EPOCHS = 20
def PreProcessImages():
	train_data = GetData(train_dir)
	test_data = GetData(test_dir)
	real_data = GetData(real_dir)

	return train_data,  test_data, real_data

def f1_metric(y_true, y_pred):
	y_true = y_true >0.4
	y_pred = y_pred>0.4
	y_true = tf.dtypes.cast(y_true,tf.float32)
	y_pred = tf.dtypes.cast(y_pred,tf.float32)

	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
	return f1_val

	
def ImportImages(train_data,  test_data):
		 
	train_dataset=tf.data.Dataset.from_tensor_slices((train_data.images, train_data.labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	test_dataset = tf.data.Dataset.from_tensor_slices((test_data.images, test_data.labels)).batch(BATCH_SIZE)
	
	return train_dataset, test_dataset


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
	
	
#https://github.com/vincent1bt/MedCycleGAN/blob/master/MedCycleGAN.ipynb	
def Discriminator():
	input = keras.layers.Input((image_size, image_size, 1), name='image')
		
	model = keras.layers.Conv2D(64, 3, strides=2, padding="same")(input)
	model = keras.layers.BatchNormalization()(model)
	model = keras.layers.LeakyReLU()(model)
  
	model = keras.layers.Conv2D(128, 3, strides=2, padding="same")(model)
	model = keras.layers.BatchNormalization()(model)
	model = keras.layers.LeakyReLU()(model)

	model = keras.layers.Conv2D(1, 2, strides=1, activation='sigmoid', padding="valid")(model)
	Model= keras.models.Model(input, model)
	return Model
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		#show_predictions()
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))\

def dice_coef(y_true, y_pred, smooth=1):
	y_true = y_true >0.4
	y_pred = y_pred>0.4
	y_true = tf.dtypes.cast(y_true,tf.float32)
	y_pred = tf.dtypes.cast(y_pred,tf.float32)

	intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
	return(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def TrainUnet():
#############################Start Unet training###############################
	epochs = 10
	batch_size = 1
	model = UNet()
	#adam = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[f1_metric,dice_coef])
	model.summary()
	
	train_data,  test_data, real_data= PreProcessImages()
	#train_dataset, test_dataset= ImportImages(train_data,  test_data)
	#print (train_data.images[0].shape)
	#train_steps = len(train_data.labels)//batch_size
	#valid_steps = len(test_data.labels)//batch_size
	#print (train_steps)
	#model.fit_generator(train_dataset, validation_data=test_dataset, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)
	if not os.path.exists("UNetW.h5"):
		model_history = model.fit(train_data.images,train_data.labels,validation_split=0.3, epochs=epochs)
		loss = model_history.history['loss']
		val_loss = model_history.history['val_loss']
	else:
		model.load_weights("UNetW.h5")
	
	resultCross = model.predict(test_data.images)

	resultSame = model.predict(real_data.images)


	resultCross = resultCross > 0.4

	resultSame = resultSame >0.4

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
		img = img>0.4
		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("SamePredicted_"+str(i)+".jpg",img)
		img = real_data.labels[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("SameGround_"+str(i)+".jpg",img)
		img = real_data.images[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("SameImage_"+str(i)+".jpg",img)
		img = resultCross[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("CrossPredicted_"+str(i)+".jpg",img)
		img = test_data.labels[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("CrossGround_"+str(i)+".jpg",img)
		img = test_data.images[i]

		img = img*255
		img = img.astype(np.uint8)
		cv2.imwrite("CrossImage_"+str(i)+".jpg",img)

	

	epochs = range(epochs)
	if not os.path.exists("UNetW.h5"):
		plt.figure()
		plt.plot(epochs, loss, 'r', label='Training loss')
		plt.plot(epochs, val_loss, 'bo', label='Validation loss')
		plt.title('Training and Validation Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss Value')
		plt.ylim([0, 1])
		plt.legend()
		plt.show()

	model.save_weights("UNetW.h5")
	return model
		
def main():
	model = TrainUnet()

if __name__ == '__main__':
	main()

