#!/usr/bin/env python

# -*- coding: utf-8 -*-

import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython import display
import pylab as pl

import glob
import imageio

def sample_dataset():
	dataset_shape = (2000, 1)
	return tf.random.normal(
		mean=10.0, shape=dataset_shape, stddev=0.1, dtype=tf.float32
	)

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
counts, bin, ignored = plt.hist(sample_dataset().numpy(), 100)
axes = plt.gca()
axes.set_xlim([-1, 11])
axes.set_ylim([0, 60])

plt.show()

def build_generator(input_shape):
	"""
	Defines the generator keras.Model.
	Args:
		input_shape: the desired input shape (e.g.: (latent_space_size))
	Returns:
		G: The generator model
	"""
	inputs = tf.keras.layers.Input(input_shape)
	net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc1")(inputs)
	net = tf.keras.layers.Dense(units=64, activation=tf.nn.elu, name="fc2")(net)
	net = tf.keras.layers.Dense(units=1, name="G")(net)
	G = tf.keras.Model(inputs=inputs, outputs=net)
	return G
	
	
def build_disciminator(input_shape):
	"""
	Defines the discriminator keras.Model.
	Args:
		input_shape: the desired input shape (e.g.: (the generator output shape))
	Returns:
		D: the discriminator model
	"""
	inputs = tf.keras.layers.Input(input_shape)
	net = tf.keras.layers.Dense(units=32, activation=tf.nn.elu, name="fc1")(inputs)
	net = tf.keras.layers.Dense(units=1, name="D")(net)
	D = tf.keras.Model(inputs=inputs, outputs=net)
	return D



# Define the real input shape, a batch of values sampled from the real data
input_shape = (1,)

# Define the discriminator model
D = build_disciminator(input_shape)

# Arbitrarily set the shape of the noise prior
latent_space_shape = (100,)

# Define the generator (along with the chosen input noise shape)
G = build_generator(latent_space_shape)	

def d_loss(real_output, generated_output):
	"""The discriminator loss function."""
	bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return bce(tf.ones_like(real_output), real_output) + bce(
		tf.zeros_like(generated_output), generated_output
	)
	
	
def g_loss(generated_output):
	"""The Generator loss function."""
	bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return bce(tf.ones_like(generated_output), generated_output)
	
	
if not os.path.exists("./gif/"):
	os.makedirs("./gif/")

# Let's play the min-max game
def train():
	# Define the optimizers and the train operations
	optimizer = tf.keras.optimizers.Adam(1e-5)

	@tf.function
	def train_step():
		with tf.GradientTape(persistent=True) as tape:
			real_data = sample_dataset()
			#real_data = ImportDataSet()
			noise_vector = tf.random.normal(mean=0, stddev=1, shape=(real_data.shape[0], latent_space_shape[0]))
			# Sample from the Generator
			fake_data = G(noise_vector)
			# Compute the D loss
			d_fake_data = D(fake_data)
			d_real_data = D(real_data)
			d_loss_value = d_loss(generated_output=d_fake_data, real_output=d_real_data)
			# Compute the G loss
			g_loss_value = g_loss(generated_output=d_fake_data)
		# Now that we have computed the losses, we can compute the gradients 
		# (using the tape) and optimize the networks
		d_gradients = tape.gradient(d_loss_value, D.trainable_variables)
		g_gradients = tape.gradient(g_loss_value, G.trainable_variables)
		del tape

		# Apply gradients to variables
		optimizer.apply_gradients(zip(d_gradients, D.trainable_variables))
		optimizer.apply_gradients(zip(g_gradients, G.trainable_variables))
		return real_data, fake_data, g_loss_value, d_loss_value

	# 40000 training steps with logging every 200 steps
	fig, ax = plt.subplots()
	for step in range(40000):
		real_data, fake_data, g_loss_value, d_loss_value = train_step()
		if step % 200 == 0:
			print(
				"G loss: ",
				g_loss_value.numpy(),
				" D loss: ",
				d_loss_value.numpy(),
				" step: ",
				step,
			)

			# Sample 5000 values from the Generator and draw the histogram
			ax.hist(fake_data.numpy(), 100)
			ax.hist(real_data.numpy(), 100)
			# these are matplotlib.patch.Patch properties
			props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

			# place a text box in upper left in axes coords
			textstr = f"step={step}"
			ax.text(
				0.05,
				0.95,
				textstr,
				transform=ax.transAxes,
				fontsize=14,
				verticalalignment="top",
				bbox=props,
			)

			axes = plt.gca()
			axes.set_xlim([-1, 11])
			axes.set_ylim([0, 60])
			display.display(pl.gcf())
			display.clear_output(wait=True)
			plt.savefig("./gif/{}.png".format(step))
			plt.gca().clear()


train()


anim_file = 'gif/learning_gaussian.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
	filenames = glob.glob('gif/*.png')
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