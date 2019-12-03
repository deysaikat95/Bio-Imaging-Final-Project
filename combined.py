import os
import random
import time
import numpy as np

import skimage.io
import cv2
#import torchvision.transforms as transforms
#from torch.utils.data import DataLoader
#from torch.autograd import Variable
from PIL import Image
#import torch

import tensorflow as tf
from tensorflow import keras
from IPython.display import clear_output
import matplotlib.pyplot as plt


#from models import Generator
#from models import Discriminator
#from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from os.path import isfile,join
from os import listdir

from tensorflow_examples.models.pix2pix import pix2pix
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
			image = cv2.imread(os.path.join(image_dir,onlyImagefiles[i]),cv2.IMREAD_GRAYSCALE)
			im = Image.open(os.path.join(label_dir,onlyLabelfiles[i]))
			label = np.array(im)

			image= cv2.resize(image, (self.image_size, self.image_size))
			label= cv2.resize(label, (self.image_size, self.image_size))
			#image = image[...,0][...,None]/255
			label = label>1
			image = image/255
			image = image[...,None]
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
BATCH_SIZE = 1
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
	Model= keras.models.Model(inputs, outputs)
	return Model
	
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		#show_predictions()
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
	
def TrainUnet():
#############################Start Unet training###############################
	epochs = 10
	batch_size = 1
	model = UNet()
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
	model.summary()
	
	train_data,  test_data= PreProcessImages()
	#train_dataset, test_dataset= ImportImages(train_data,  test_data)
	#print (train_data.images[0].shape)
	#train_steps = len(train_data.labels)//batch_size
	#valid_steps = len(test_data.labels)//batch_size
	#print (train_steps)
	#model.fit_generator(train_dataset, validation_data=test_dataset, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)
	model_history = model.fit(test_data.images,test_data.labels, epochs=10)
	loss = model_history.history['loss']

	score = model.evaluate(train_data.images,train_data.labels)
	print(score[0])
	print(score[1])
'''
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
	'''

def TrainGAN():
#############################Start GAN training########################################

###### Definition of variables ######
# Networks
	netG_A2B = Generator(opt.input_nc, opt.output_nc)
	netG_B2A = Generator(opt.output_nc, opt.input_nc)
	netD_A = Discriminator(opt.input_nc)
	netD_B = Discriminator(opt.output_nc)

	if opt.cuda:
		netG_A2B.cuda()
		netG_B2A.cuda()
		netD_A.cuda()
		netD_B.cuda()

	netG_A2B.apply(weights_init_normal)
	netG_B2A.apply(weights_init_normal)
	netD_A.apply(weights_init_normal)
	netD_B.apply(weights_init_normal)

	# Lossess
	criterion_GAN = torch.nn.MSELoss()
	criterion_cycle = torch.nn.L1Loss()
	criterion_identity = torch.nn.L1Loss()

	# Optimizers & LR schedulers
	optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
										lr=opt.lr, betas=(0.5, 0.999))
	optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
	optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

	lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
	lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
	lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

	# Inputs & targets memory allocation
	Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
	input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
	input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
	target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
	target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

	fake_A_buffer = ReplayBuffer()
	fake_B_buffer = ReplayBuffer()

	# Dataset loader
	transforms_ = [ transforms.Resize(int(opt.size*1.12), Image.BICUBIC), 
				transforms.RandomCrop(opt.size), 
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
	dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
						batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

	# Loss plot
	logger = Logger(opt.n_epochs, len(dataloader))
	###################################

	###### Training ######
	for epoch in range(opt.epoch, opt.n_epochs):
		for i, batch in enumerate(dataloader):
			# Set model input
			real_A = Variable(input_A.copy_(batch['A']))
			real_B = Variable(input_B.copy_(batch['B']))

			###### Generators A2B and B2A ######
			optimizer_G.zero_grad()

			# Identity loss
			# G_A2B(B) should equal B if real B is fed
			same_B = netG_A2B(real_B)
			loss_identity_B = criterion_identity(same_B, real_B)*5.0
			# G_B2A(A) should equal A if real A is fed
			same_A = netG_B2A(real_A)
			loss_identity_A = criterion_identity(same_A, real_A)*5.0

			# GAN loss
			fake_B = netG_A2B(real_A)
			pred_fake = netD_B(fake_B)
			loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

			fake_A = netG_B2A(real_B)
			pred_fake = netD_A(fake_A)
			loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

			# Cycle loss
			recovered_A = netG_B2A(fake_B)
			loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

			recovered_B = netG_A2B(fake_A)
			loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

			# Total loss
			loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
			loss_G.backward()

			optimizer_G.step()
			###################################

			###### Discriminator A ######
			optimizer_D_A.zero_grad()

			# Real loss
			pred_real = netD_A(real_A)
			loss_D_real = criterion_GAN(pred_real, target_real)

			# Fake loss
			fake_A = fake_A_buffer.push_and_pop(fake_A)
			pred_fake = netD_A(fake_A.detach())
			loss_D_fake = criterion_GAN(pred_fake, target_fake)

			# Total loss
			loss_D_A = (loss_D_real + loss_D_fake)*0.5
			loss_D_A.backward()

		optimizer_D_A.step()
		###################################

		###### Discriminator B ######
		optimizer_D_B.zero_grad()

		# Real loss
		pred_real = netD_B(real_B)
		loss_D_real = criterion_GAN(pred_real, target_real)

		# Fake loss
		fake_B = fake_B_buffer.push_and_pop(fake_B)
		pred_fake = netD_B(fake_B.detach())
		loss_D_fake = criterion_GAN(pred_fake, target_fake)

		# Total loss
		loss_D_B = (loss_D_real + loss_D_fake)*0.5
		loss_D_B.backward()

		optimizer_D_B.step()
		###################################

		# Progress report (http://localhost:8097)
		logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
					'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
					images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

	# Update learning rates
	lr_scheduler_G.step()
	lr_scheduler_D_A.step()
	lr_scheduler_D_B.step()

	# Save models checkpoints
	torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
	torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
	torch.save(netD_A.state_dict(), 'output/netD_A.pth')
	torch.save(netD_B.state_dict(), 'output/netD_B.pth')

	


# Generators and Discriminators 
generator_1 = unet()
generator_2 = unet()
discriminator_x = Discriminator()
discriminator_y = Discriminator()

# Loss Calculations 
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Discriminator Loss 
def Discriminator_Loss(real, gen):
	real_loss = cross_entropy(tf.ones_like(real), real)
	generated_loss = cross_entropy(tf.zeros_like(gen), gen)
	
	loss = (real_loss + generated_loss)* 0.5
	
	return loss

# Generator Loss
def Generator_Loss(gen):
	gen_loss = cross_entropy(tf.ones_like(gen), gen)
	
	return gen_loss
	
# Cycle GAN Loss 
Lambda = 10
def Cycle_Loss(real_image, cycle_image):
	loss_L1= tf.reduce_mean(tf.abs(real_image-cycle_image))
	Loss = Lambda * loss_L1
	return Loss
	
# Identity Loss 
def Identity_Loss(real_image, same_image):
	loss = tf.reduce_mean(tf.abs(real_image-same_image))
	Loss = 0.5 * Lambda * loss
	return Loss


# Optimizers

generator_1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_2_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Training 
# Generating an image
def generate_images(model, test_input):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Real Image', 'Generated Image']

  for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(title[i])
    plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5, cmap='gray')
    plt.axis('off')
  plt.show()

#Steps of Training
@tf.function
def train_step(real_image_x, real_image_y):
  with tf.GradientTape(persistent=True) as tape: 
    #generator_g transforms from X to Y
    #generator_f transforms from Y to X

    #From X, generate Y
    generated_image_y = generator_g(real_image_x, training=True)
    #From generated Y, back to X
    cycled_image_x = generator_f(generated_image_y, training=True)

    #From Y, generate X
    generated_image_x = generator_f(real_image_y, training=True)
    #From generated X, back to Y
    cycled_image_y = generator_g(generated_image_x, training=True)

    same_image_x = generator_f(real_image_x, training=True)
    same_image_y = generator_g(real_image_y, training=True)

    #Should output 1's since are real images
    patch_real_x = discriminator_x(real_image_x, training=True)
    patch_real_y = discriminator_y(real_image_y, training=True)

    #Should output 0's since are generated images
    patch_generated_x = discriminator_x(generated_image_x, training=True)
    patch_generated_y = discriminator_y(generated_image_y, training=True)

    generator_g_loss = generator_loss(patch_generated_y)
    generator_f_loss = generator_loss(patch_generated_x)
    
    total_cycle_loss = cycle_loss(real_image_x, cycled_image_x) + cycle_loss(real_image_y, cycled_image_y)
    
    total_generator_g_loss = generator_g_loss + total_cycle_loss + identity_loss(real_image_y, same_image_y)
    total_generator_f_loss = generator_f_loss + total_cycle_loss + identity_loss(real_image_x, same_image_x)

    discriminator_x_loss = discriminator_loss(patch_real_x, patch_generated_x)
    discriminator_y_loss = discriminator_loss(patch_real_y, patch_generated_y)
  
  generator_g_gradients = tape.gradient(total_generator_g_loss, generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_generator_f_loss, generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(discriminator_x_loss, discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(discriminator_y_loss, discriminator_y.trainable_variables)
  
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

def Train(epochs):
	for epoch in range(epochs):
		start= time.time()
		
		n = 0
		for image_x, image_y in tf.data.Dataset.zip((mr_train_dataset, ct_train_dataset)):
      train_step(image_x, image_y)
      
      if n % 10 == 0:
        print ('.', end='')
      n+=1

    display.clear_output(wait=True)

    for test_image_x in mr_test_dataset.take(5):
      generate_images(generator_g, test_image_x)

    print('Time taken for epoch {} was {} sec\n'.format(epoch + 1, time.time() - start))

  checkpoint.save(file_prefix=checkpoint_prefix)
		
def main():
	#TrainUnet()
	Train(5)


if __name__ == '__main__':
	main()

