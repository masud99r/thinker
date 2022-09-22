import argparse
import os, sys
import numpy as np
import math
import itertools
import datetime
import time
import random

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from PIL import Image

from stargan_models import *

# from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# import torch.multiprocessing as mp
# mp.set_start_method('spawn')



from os.path import join, exists
from os import mkdir

class ImageDataset(Dataset):
	def __init__(self, data, labels, transforms_=None):
		self.transform = transforms.Compose(transforms_)
		# print('labels.shape', labels.shape)
		# print('data.shape', data.shape)
		# print('data.type', type(data))
		assert data.shape[0] == len(labels) # each image has to associate with one label
		self.data = data
		self.labels = labels

		# print('self.labels', self.labels)
		
	def __getitem__(self, index):
		i = index % len(self.data)
		image = self.data[i]
		label = self.labels[i]

		
		# print('label inside getitem', label)
		image = image.astype(np.uint8)
		# label = label.astype(np.uint8)

		image = torch.from_numpy(image)
		image = self.transform(image)

		label = torch.FloatTensor(np.array(label))
		
		# print('label', label)
		# print('label.type', type(label))
		# print('image', image)
		# print('image.type', type(image))
		return image, label
		# return {"image": image, "label": label}

	def __len__(self):
		return len(self.data)

class StarGAN():
	def __init__(self, gan_dir, 
				epoch=0, 
				n_epochs=False, 
				batch_size=16, 
				lr=0.0002, 
				b1=0.5, 
				b2=0.999, 
				n_cpu=1, 
				img_height=64, 
				img_width=64, 
				channels=3, 
				sample_interval=10, 
				checkpoint_interval=1, 
				n_residual_blocks=6, 
				selected_attrs=['0', '1', '2'],
				n_critic=5, 
				device='cpu'):
		
		self.gan_dir = gan_dir
		self.epoch=epoch
		self.n_epochs=n_epochs
		self.batch_size=batch_size
		self.lr=lr
		self.b1=b1
		self.b2=b2
		self.n_cpu=n_cpu
		self.img_height=img_height
		self.img_width=img_width
		self.channels=channels
		self.sample_interval=sample_interval 
		self.checkpoint_interval=checkpoint_interval
		self.n_residual_blocks=n_residual_blocks
		self.selected_attrs=selected_attrs
		self.n_critic=n_critic
		self.device = device

		# Loss weights
		self.lambda_cls = 1
		self.lambda_rec = 10
		self.lambda_gp = 10

		self.c_dim = len(self.selected_attrs)

		# Image transformations
		# self.transforms_ = [
		# 	transforms.ToPILImage(),
		# 	transforms.Resize(int(self.img_height * 1.12), Image.BICUBIC),
		# 	transforms.RandomCrop((self.img_height, self.img_width)),
		# 	transforms.RandomHorizontalFlip(),
		# 	transforms.ToTensor(),
		# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		# ]
		self.train_transforms_ = [
					transforms.ToPILImage(),
					transforms.Resize(int(1.12 * self.img_height), Image.BICUBIC),
					transforms.RandomCrop(self.img_height, self.img_width),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				]
		self.inf_transforms_ = [
						transforms.ToPILImage(),
						transforms.Resize((self.img_height, self.img_width), Image.BICUBIC),
						transforms.ToTensor(),
						transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					]
		# self.train_transforms = transforms.Compose(self.train_transforms_)
		self.inf_transforms = transforms.Compose(self.inf_transforms_)
		
		self.train_count = 0

		self.label_changes = [
			((0, 1), (0, 2)),  # 
			((1, 0), (1, 2)),  # 
			((2, 0), (2, 1)),  # 
		]	
		# self.init_generator()
		# initt training
	def criterion_cls(self, logit, target):
		return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
	def init_generator(self):
		self.gan_dir_saved = join(self.gan_dir, 'saved_cgan')
		if not exists(self.gan_dir_saved):
			mkdir(self.gan_dir_saved)
			mkdir(join(self.gan_dir_saved, 'images'))
			mkdir(join(self.gan_dir_saved, 'saved_models'))

		# Loss functions
		self.criterion_cycle = torch.nn.L1Loss().to(self.device)

		input_shape = (self.channels, self.img_height, self.img_width)
		# Initialize generator and discriminator
		self.generator = GeneratorResNet(img_shape=input_shape, res_blocks=self.n_residual_blocks, c_dim=self.c_dim).to(self.device)
		self.discriminator = Discriminator(img_shape=input_shape, c_dim=self.c_dim).to(self.device)
		
		if self.epoch != 0:
			# Load pretrained models
			self.generator.load_state_dict(torch.load(self.gan_dir_saved+"saved_models/generator.pth"))
			self.discriminator.load_state_dict(torch.load(self.gan_dir_saved+"saved_models/discriminator.pth"))
		else:
			self.generator.apply(weights_init_normal)
			self.discriminator.apply(weights_init_normal)

		# Optimizers
		self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
		self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

				# Training data loader
		# self.dataloader = DataLoader(
		# 	ImageDataset(data=data, labels=labels, transforms_=self.train_transforms_),
		# 	batch_size=self.batch_size,
		# 	shuffle=True,
		# 	num_workers=self.n_cpu,
		# )
		# self.val_dataloader = DataLoader(
		# 	ImageDataset(data=data, labels=labels, transforms_=self.inf_transforms_),
		# 	batch_size=10,
		# 	shuffle=True,
		# 	num_workers=self.n_cpu,
		# )
	def load_generator(self):
		self.gan_dir_saved = join(self.gan_dir, 'saved_cgan')
		# if not exists(self.gan_dir_saved):
		# 	mkdir(self.gan_dir_saved)
		# 	mkdir(join(self.gan_dir_saved, 'images'))
		# 	mkdir(join(self.gan_dir_saved, 'saved_models'))

		# Loss functions
		self.criterion_cycle = torch.nn.L1Loss().to(self.device)

		input_shape = (self.channels, self.img_height, self.img_width)
		# Initialize generator and discriminator
		self.generator = GeneratorResNet(img_shape=input_shape, res_blocks=self.n_residual_blocks, c_dim=self.c_dim).to(self.device)
		self.discriminator = Discriminator(img_shape=input_shape, c_dim=self.c_dim).to(self.device)
		
		# if self.epoch != 0:
		# 	# Load pretrained models
		self.generator.load_state_dict(torch.load(self.gan_dir_saved+"/saved_models/generator.pth"))
		self.discriminator.load_state_dict(torch.load(self.gan_dir_saved+"/saved_models/discriminator.pth"))
		
	def compute_gradient_penalty(self, D, real_samples, fake_samples):
		if self.device == 'cpu':
			Tensor = torch.Tensor
		else: 
			Tensor = torch.cuda.FloatTensor
		"""Calculates the gradient penalty loss for WGAN GP"""
		# Random weight term for interpolation between real and fake samples
		alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
		# Get random interpolation between real and fake samples
		interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
		d_interpolates, _ = D(interpolates)
		fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
		# Get gradient w.r.t. interpolates
		gradients = autograd.grad(
			outputs=d_interpolates,
			inputs=interpolates,
			grad_outputs=fake,
			create_graph=True,
			retain_graph=True,
			only_inputs=True,
		)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return gradient_penalty		
	def train(self, data, labels, n_iteration=1):
		data = data.permute(0, 3, 1, 2)
		data = data.to('cpu').numpy()
		# print('labels in gen', labels)
		if self.device == 'cpu':
			Tensor = torch.Tensor
		else: 
			Tensor = torch.cuda.FloatTensor
		self.train_count += 1

		# def collate_fn(batch):
		# 	return tuple(zip(*batch))

		dataloader = DataLoader(
			ImageDataset(data=data, labels=labels, transforms_=self.train_transforms_),
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.n_cpu,
			# collate_fn=collate_fn,
		)
		val_dataloader = DataLoader(
			ImageDataset(data=data, labels=labels, transforms_=self.inf_transforms_),
			batch_size=10,
			shuffle=True,
			num_workers=self.n_cpu,
			# collate_fn=collate_fn,
		)



		def sample_images(batches_done):
			"""Saves a generated sample of domain translations"""
			val_imgs, val_labels = next(iter(val_dataloader))
			val_imgs = Variable(val_imgs.type(Tensor))
			val_labels = Variable(val_labels.type(Tensor))
			img_samples = None
			for i in range(10):
				img, label = val_imgs[i], val_labels[i]
				# Repeat for number of label changes
				imgs = img.repeat(self.c_dim, 1, 1, 1)
				labels = label.repeat(self.c_dim, 1)
				# Make changes to labels
				for sample_i, changes in enumerate(self.label_changes):
					for col, val in changes:
						labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

				# Generate translations
				gen_imgs = self.generator(imgs, labels)
				# Concatenate images by width
				gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
				img_sample = torch.cat((img.data, gen_imgs), -1)
				# Add as row to generated samples
				img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

			save_image(img_samples.view(1, *img_samples.shape), self.gan_dir_saved+"/images/%s.png" % batches_done, normalize=True)



		prev_time = time.time()
		for epoch in range(n_iteration):
			for i, batch in enumerate(dataloader):
			# for bi, batch in enumerate(dataloader):
				# Model inputs
				imgs, labels = batch
				# print('imgs value', imgs)
				# print('imgslen', len(imgs))
				# print(i, 'labels', labels)
				# print('imgs.type', type(imgs))
				imgs = torch.stack(list(imgs), dim=0)
				# print('imgs.shape', imgs.shape)


				labels = torch.stack(list(labels), dim=0)


				imgs = Variable(imgs.type(Tensor))
				labels = Variable(labels.type(Tensor))

				# imgs = Variable(batch["image"].type(Tensor))
				# labels = Variable(batch["label"].type(Tensor))

				# Sample labels as generator inputs
				sampled_c = Variable(Tensor(np.random.randint(0, 2, (imgs.size(0), self.c_dim))))
				# Generate fake batch of images
				fake_imgs = self.generator(imgs, sampled_c)

				# ---------------------
				#  Train Discriminator
				# ---------------------

				self.optimizer_D.zero_grad()

				# Real images
				real_validity, pred_cls = self.discriminator(imgs)
				# Fake images
				fake_validity, _ = self.discriminator(fake_imgs.detach())
				# Gradient penalty
				gradient_penalty = self.compute_gradient_penalty(self.discriminator, imgs.data, fake_imgs.data)
				# Adversarial loss
				loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
				# Classification loss
				loss_D_cls = self.criterion_cls(pred_cls, labels)
				# Total loss
				loss_D = loss_D_adv + self.lambda_cls * loss_D_cls

				loss_D.backward()
				self.optimizer_D.step()

				self.optimizer_G.zero_grad()

				# Every n_critic times update generator
				if i % self.n_critic == 0:

					# -----------------
					#  Train Generator
					# -----------------

					# Translate and reconstruct image
					gen_imgs = self.generator(imgs, sampled_c)
					recov_imgs = self.generator(gen_imgs, labels)
					# Discriminator evaluates translated image
					fake_validity, pred_cls = self.discriminator(gen_imgs)
					# Adversarial loss
					loss_G_adv = -torch.mean(fake_validity)
					# Classification loss
					loss_G_cls = self.criterion_cls(pred_cls, sampled_c)
					# Reconstruction loss
					loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
					# Total loss
					loss_G = loss_G_adv + self.lambda_cls * loss_G_cls + self.lambda_rec * loss_G_rec

					loss_G.backward()
					self.optimizer_G.step()

					# --------------
					#  Log Progress
					# --------------

					# Determine approximate time left
					# batches_done = epoch * len(self.dataloader) + i
					# batches_left = self.n_epochs * len(dataloader) - batches_done
					# time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

					# Print log
					# print(epoch,
					# 		loss_D_adv.item(),
					# 		loss_D_cls.item(),
					# 		loss_G.item(),
					# 		loss_G_adv.item(),
					# 		loss_G_cls.item(),
					# 		loss_G_rec.item())

					# If at sample interval sample and save image
					if epoch % self.sample_interval == 0:
						sample_images(self.train_count* len(dataloader)+ epoch)

			if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
				# Save model checkpoints
				# torch.save(self.generator.state_dict(), self.gan_dir_saved+"/saved_models/generator_%d.pth" % epoch)
				# torch.save(self.discriminator.state_dict(), self.gan_dir_saved+"/saved_models/discriminator_%d.pth" % epoch)
				torch.save(self.generator.state_dict(), self.gan_dir_saved+"/saved_models/generator.pth")
				torch.save(self.discriminator.state_dict(), self.gan_dir_saved+"/saved_models/discriminator.pth")
			
			else:
				print('Not saving')
	
	def convert_dist(self, obs, target_dist):
		
		obs = obs.permute(0, 3, 1, 2) # from _, 64, 64, 3 to _, 3, 64, 64
		obs_trans = torch.zeros(obs.shape[0], 3, 64, 64)
		
		for i, x in enumerate(obs): 
			x = self.inf_transforms(x)
			obs_trans[i] = x
		# target_dist	
		# obs = obs_trans.
		# # Repeat for number of label changes
		# imgs = img.repeat(self.c_dim, 1, 1, 1)
		# labels = label.repeat(self.c_dim, 1)
		# val_imgs = Variable(val_imgs.type(Tensor))
		# labels = Variable(target_dist.type(Tensor))
		if self.device == 'cpu':
			obs_trans = obs_trans.type(torch.FloatTensor)
		else: 
			obs_trans = obs_trans.type(torch.cuda.FloatTensor)

		obs_trans = self.generator(obs_trans, target_dist)
		obs_trans = obs_trans.permute(0, 2, 3, 1) # back to expected shape
		
		return obs_trans