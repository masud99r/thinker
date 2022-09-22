import torch
from sklearn import mixture
from joblib import dump, load
import numpy as np


class Cluster():
	def __init__(self, cluster_dir, load_model=False, n_components=2, device='cpu'):
		
		self.device = device
		#https://github.com/pytorch/pytorch/issues/61755#issuecomment-885801511
		torch.hub._validate_not_a_forked_repo=lambda a,b,c: True 
		self.model_resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
		self.model_resnet.eval()
		self.model_resnet.to(device)
		self.model_gmm = None
		self.n_components = n_components

		self.cluster_dir = cluster_dir
		self.load_model = load_model
	
	def train(self, data):
		# filename = self.cluster_dir+'gmm.joblib'
		if self.load_model:
			print('GMM Loading Trained Model')
			self.model_gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type='full', verbose=0, tol=1e-4)
			self.model_gmm = load(self.cluster_dir+'gmm.joblib') 
		else:
			data = data.to(self.device)
			data = data.permute(0, 3, 1, 2)
			with torch.no_grad():
				# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
				output = self.model_resnet(data)	
			output = output.to('cpu')
			X_train = output.detach().numpy()
		
			print('GMM training start')
			self.model_gmm = mixture.GaussianMixture(n_components=self.n_components, covariance_type='full', verbose=0, tol=1e-4)
			self.model_gmm.fit(X_train)
			
			# pickle.dump(self.model_gmm, open(filename, 'wb'))
			print('Dumping training start', self.cluster_dir+'gmm.joblib')
			dump(self.model_gmm, self.cluster_dir+'gmm.joblib') 
			print('GMM Dumped at', self.cluster_dir+'gmm.joblib')	
	
	def get_labels(self, obs):
		obs = obs.permute(0, 3, 1, 2)
		obs = obs.to(self.device)
		obs = self.model_resnet(obs)
		obs = obs.to('cpu')
		obs = obs.detach().numpy()
		prob = self.model_gmm.predict_proba(obs)
		c_pred = np.argmax(prob, axis=1)
		return c_pred
	
	def get_cluster_labels(self, data):
		data = data.to(self.device)
		data = data.permute(0, 3, 1, 2)
		with torch.no_grad():
			# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
			output = self.model_resnet(data)	
		output = output.to('cpu')
		output = output.detach().numpy()

		# print('Inference')
		data = data.to('cpu')
		prob = self.model_gmm.predict_proba(output)
		c_pred = np.argmax(prob, axis=1)
		# C = []
		# for _ in range(self.n_components):
		# 	C.append([])
		# c_pred = np.argmax(prob, axis=1)
		# for j in range(c_pred.shape[0]):
		# 	C[c_pred[j]].append(data[j].numpy())
		
		return c_pred
	
	
