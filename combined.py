import os
import random

import numpy as np

import skimage.io
import cv2

import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt
# Updated to work  with tensorflow 2.0
class GetData():
	def __init__(self, data_dir):	
		images_list =[]
		
		labels_list = []		
		self.source_list = []
		
		self.image_size = 128
		examples = 0
		print("loading images")
		label_dir = os.path.join(data_dir, "Labels")
		image_dir = os.path.join(data_dir, "Images")
		for label_root, dir, files in os.walk(label_dir):
			for file in files:
				if not file.endswith((".png", ".jpg", ".gif")):
					continue
				try:
					folder = os.path.relpath(label_root,	label_dir)
					image_root = os.path.join(image_dir, folder)


					#image = cv2.imread(os.path.join(image_root, file))
					#label = cv2.imread(os.path.join(label_root, file),-1)

					#image= cv2.resize(image, (self.image_size, self.image_size))
					#label= cv2.resize(label, (self.image_size, self.image_size))
					#label=np.expand_dims(label, axis=-1)
					#image=image/255.0
					#label=label/255.0
					
					image = cv2.imread(os.path.join(image_root,file))
					label = cv2.imread(os.path.join(label_root, file))
					image= cv2.resize(image, (self.image_size, self.image_size))
					label= cv2.resize(label, (self.image_size, self.image_size))
					image = image[...,0][...,None]/255
					label = label[...,0]>1
					label = label[...,None]
					label = label.astype(np.int32)
					images_list.append(image)
					labels_list.append(label)
					examples = examples + 1
				except Exception as e:
					print(e)
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
BATCH_SIZE = 8
BUFFER_SIZE = 1000
image_size = 128
EPOCHS = 20
def PreProcessImages():
    train_data = GetData(train_dir)
    test_data = GetData(test_dir)
    return train_data,  test_data


	
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
    inputs = keras.layers.Input((image_size, image_size, 1))
    
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
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
	
	
	
def main():

	epochs = 5
	batch_size = 64
	model = UNet()
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
	model.summary()
	
	train_data,  test_data= PreProcessImages()
	train_dataset, test_dataset= ImportImages(train_data,  test_data)
	print (train_data.images[0].shape)
	train_steps = len(train_data.labels)//batch_size
	valid_steps = len(test_data.labels)//batch_size
	print (train_steps)
	#model.fit_generator(train_dataset, validation_data=test_dataset, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)
	model_history = model.fit(train_dataset, epochs=epochs,
                          steps_per_epoch=train_steps,
                          validation_steps=valid_steps,
                          validation_data=test_dataset, callbacks=[DisplayCallback()])    
	loss = model_history.history['loss']
	val_loss = model_history.history['val_loss']

	epochs = range(epochs)

	plt.figure()
	plt.plot(epochs, loss, 'r', label='Training loss')
	plt.plot(epochs, val_loss, 'bo', label='Validation loss')
	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Value')
	plt.ylim([0, 1])
	plt.legend()
	plt.show()

if __name__ == '__main__':
    main()