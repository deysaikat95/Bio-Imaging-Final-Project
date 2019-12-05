#!/usr/bin/env python

# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

# Trying to implement a cycle  Adversarial Network cGAN, 
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os, time
import PIL
import tensorflow as tf
# Added for simplicity 
from tensorflow.keras import layers

import IPython
from IPython import display # use to display
# Cross-entropy
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])




#https://hardikbansal.github.io/CycleGANBlog/
# Helper Functions 
# Convolution layer 
def ImportDataSet():
	(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
	train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
	train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
	
	BUFFER_SIZE = 60000
	BATCH_SIZE = 256
	
	# Batch and shuffle the data
	train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	return train_dataset
#def ConvLayer():

# GIF maker to save images as gis
def generate_and_save(model,epoch, test_input):
	predictions= model(test_input, training=False)
	
	fig = plt.figure(figsize=(4,4))
	
	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
		plt.axis('off')

	plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
	#plt.show() 


# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

# Generator 
def Generator():

	"""
	Generator has 4 convolution Layers all followed by a BN except the last one
	Rectified Linear Unit (ReLU) activation 
	"""

	# First  project and reshape
	model = tf.keras.Sequential()
	model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	
	model.add(layers.Reshape((7, 7, 256)))
	assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

	
	# Add The first 2D convolution  layer 
	model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
	assert model.output_shape == (None, 7, 7, 128)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	
	# Add 2nd layer of 2D convolution 
	model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
	assert model.output_shape == (None, 14, 14, 64)
	model.add(layers.BatchNormalization())
	model.add(layers.LeakyReLU())
	
	# Add the final 2D convolution layer 
	model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
	assert model.output_shape == (None, 28, 28, 1)


	return model
	
# Discriminator 
def Discriminator():

	# CNN-based image classifier
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',input_shape=[28, 28, 1]))
	
	# convolution layers
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
	model.add(layers.LeakyReLU())
	model.add(layers.Dropout(0.3))

	model.add(layers.Flatten())
	model.add(layers.Dense(1))
	
	return model

# Loss functions 
# Generator loss
def Generator_Loss(fake_out):
	return cross_entropy(tf.ones_like(fake_out),fake_out)

# Discriminator Loss
def Discriminator_Loss(real_out, fake_out):
	real_loss= cross_entropy(tf.ones_like(real_out),real_out)
	fake_loss = cross_entropy(tf.ones_like(fake_out),fake_out)
	total_loss = real_loss + fake_loss
	return total_loss

generator = Generator()
discriminator = Discriminator()
# Set up Check point saving 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def Train_Step(images):
	noise= tf.random.normal([BATCH_SIZE, noise_dim])
	
	with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
		generated_image = generator(noise, training = True)
		
		real_out = discriminator(images, training = True)
		fake_out = discriminator(generated_image, training = True) 
		
		gen_los = Generator_Loss(fake_out)
		disc_los = Discriminator_Loss(real_out, fake_out)
		
	grads_of_Gen= gen_tape.gradient(gen_los, generator.trainable_variables)
	grads_of_Disc =disc_tape.gradient(disc_los, discriminator.trainable_variables) 
	
	generator_optimizer.apply_gradients(zip(grads_of_Gen, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(grads_of_Disc, discriminator.trainable_variables))
		
# Training Model		
def Train(dataset, epochs):
	
	
	for epoch in range(epochs):
		start = time.time() # Track time 
		
		for image_batch in dataset:
			Train_Step(image_batch)
		# Making Gifs along the way 
		display.clear_output(wait=True)
		generate_and_save(generator, epoch + 1, seed)
		
		# save the model 
		if (epoch + 1) % 15 == 0:
			checkpoint.save(file_prefix = checkpoint_prefix) 
		print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
		
	# Generate after the final epoch
	display.clear_output(wait=True)
	generate_and_save(generator,
                           epochs,
                           seed)


def main():	

	noise = tf.random.normal([1, 100])
	generated_image = generator(noise, training=False)

	plt.imshow(generated_image[0, :, :, 0], cmap='gray')
	plt.show()
	print('map')
	
	
	
	decision = discriminator(generated_image)
	print(decision)
	data= ImportDataSet()
	Train(data, EPOCHS)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
	display_image(EPOCHS)
	
	
	anim_file = 'dcgan.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
		filenames = glob.glob('image*.png')
		filenames = sorted(filenames)
		last = -1
		for i,filename in enumerate(filenames):
			frame = 2*(i**0.5)
			if round(frame) > round(last):
				last = frame
			else:
				continue
			image = imageio.imread(filename)
			writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)
	
	display.Image(filename=anim_file)

if __name__ == '__main__':
	main()