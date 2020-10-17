from utils.model_utils import *
from utils.bag_utils import *
from utils.data_utils import *

import torch
from torch.utils import data
from torch.autograd import Function

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

import numpy as np

import itertools
import os

def blockPrint():
	sys.stdout = open(os.devnull, 'w')

def enablePrint():
	sys.stdout = sys.__stdout__

class Dataset(data.Dataset):

	def __init__(self, X, y, eta_plus=None, eta_minus=None, omega=None):
		self.X = torch.from_numpy(X).float()
		self.y = torch.from_numpy(y).float()
		self.eta_plus = None
		self.eta_minus = None
		self.omega = None
		if eta_plus is not None:
			self.eta_plus = torch.from_numpy(eta_plus).float()
		if eta_minus is not None:
			self.eta_minus = torch.from_numpy(eta_minus).float()
		if omega is not None:
			self.omega = torch.from_numpy(omega).float()

	def __len__(self):
		return self.y.shape[0]

	def __getitem__(self, index):
		if (self.eta_plus is not None) and (self.omega is not None) and (self.eta_minus is not None):  # for corrected loss
			return self.X[index, :], self.y[index], self.eta_plus[index], self.eta_minus[index], self.omega[index]
		return self.X[index, :], self.y[index]  # for full supervised case

class LogisticLossHelper():

	@staticmethod
	def forward(input, target):
		return torch.log(1 + torch.exp(-target*input))

	@staticmethod
	def backward(input, target):
		return (-target) * torch.div(1, 1 + torch.exp(target * input))


def precomputation(train_X, bag_id, prop_dict, size_dict, weighting_strategy='fixed_size_optimal',
				   pairing_strategy='fixed_size_optimal' ,seed=None, verbose=1):
	pairs, bag_to_pair = generate_pairs(set(bag_id), prop_dict, seed=seed, strategy=pairing_strategy)
	enablePrint() #always print the invalid bags situation
	train_X, bag_id, pairs, _ = drop_invalid_bags(train_X, bag_id, pairs, bag_to_pair)
	if not verbose:
		blockPrint()
	print("valid bags: {:d}".format(len(set(bag_id))))
	weights = assign_weights(pairs, strategy=weighting_strategy)
	pair_to_size = pair_size(size_dict, pairs)
	noisy_y = recover_noisy_labels(bag_id, bag_to_pair, pairs)
	eta_plus, eta_minus = compute_eta(np.vectorize(bag_to_pair.get)(bag_id), pairs, noisy_y)
	omega = compute_omega(np.vectorize(bag_to_pair.get)(bag_id), weights, pair_to_size)
	return train_X, noisy_y, eta_plus, eta_minus, omega


def correct_loss(Loss):

	class CorrectedLoss(Function):

		@staticmethod
		def forward(ctx, input, eta_plus, eta_minus, omega, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
			ctx.save_for_backward(input, eta_plus, eta_minus, omega)
			ones = torch.ones(input.size(), dtype=torch.double).to(device)
			neg_ones = -torch.ones(input.size(), dtype=torch.double).to(device)
			return (omega * (eta_plus * Loss.forward(input, ones) + eta_minus * Loss.forward(input, neg_ones))).sum()

		@staticmethod
		def backward(ctx, grad_output, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
			input, eta_plus, eta_minus, omega = ctx.saved_tensors
			ones = torch.ones(input.size(), dtype=torch.double).to(device)
			neg_ones = -torch.ones(input.size(), dtype=torch.double).to(device)
			epsilon_plus = Loss.backward(input, ones)
			epsilon_minus = Loss.backward(input, neg_ones)
			grad_input = omega * (eta_plus * epsilon_plus + eta_minus * epsilon_minus)
			grad_input *= grad_output
			return grad_input, None, None, None

	return CorrectedLoss

class NN_Model():
	# an interface to use cross validation
	def __init__(self, network, loss, test_criterion, optimizer, max_epoch, device,
				 score_to_label=lambda x: ((x > 0).astype(int) * 2 - 1)):
		self.model = network
		self.train_criterion = correct_loss(loss).apply
		self.test_criterion = test_criterion
		self.optimizer = optimizer
		self.max_epoch = max_epoch
		self.device = device
		self.model = self.model.to(self.device)
		self.score_to_label=score_to_label

		self.model.__init__()
		self.model = self.model.to(self.device)
		self.optimizer.__init__(self.model.parameters(), **self.optimizer.defaults)

	def fit(self, train_X, val_X, val_y, train_bags, prop_dict, size_dict, batch_size, log_interval, reinitialize=0, verbose=1,
			weighting_strategy='fixed_size_optimal', pairing_strategy='fixed_size_optimal', min_epoch=50, stop_criterion=1e-8):
		self.weighting_strategy = weighting_strategy
		self.pairing_strategy = pairing_strategy
		if reinitialize:
			self.model.__init__()
			self.model = self.model.to(self.device)
			self.optimizer.__init__(self.model.parameters(), **self.optimizer.defaults)
		if not verbose:
			blockPrint()
		train_X = pd_to_numpy(train_X)
		val_X = pd_to_numpy(val_X)
		val_y = pd_to_numpy(val_y)
		train_X, noisy_y, eta_plus, eta_minus, omega = precomputation(train_X, train_bags, prop_dict, size_dict,
																	  verbose=verbose,
																	  weighting_strategy=weighting_strategy,
																	  pairing_strategy=pairing_strategy)
		train_set = Dataset(train_X, noisy_y, eta_plus=eta_plus, eta_minus=eta_minus, omega=omega)
		val_set = Dataset(val_X, val_y)
		params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4}
		train_loader = data.DataLoader(train_set, **params)
		test_loader = data.DataLoader(val_set, **params)
		prev_loss = None
		for epoch in range(1, self.max_epoch + 1):
			epoch_loss = self._train_setp(train_loader, log_interval, epoch)
			if epoch % log_interval == 0:
				pred = self.predict(val_X)
				print("validation accuracy = ", self.get_accuracy(val_X, val_y, prediction=pred))
				print("balanced validation accuracy = ", self.get_balanced_accuracy(val_X, val_y, prediction=pred))
			if prev_loss is None:
				prev_loss = epoch_loss
			elif abs(prev_loss - epoch_loss) < stop_criterion and epoch > min_epoch:
				break
			else:
				prev_loss = epoch_loss
		if not verbose:
			enablePrint()

	def _train_setp(self, train_loader, log_interval, epoch):
		self.model.train()
		total_loss = 0.0
		for batch_idx, (data, target, eta_plus, eta_minus, omega) in enumerate(train_loader):
			data, target, eta_plus, eta_minus, omega = data.to(self.device), target.double().to(self.device), \
													   eta_plus.double().to(self.device), \
													   eta_minus.double().to(self.device), \
													   omega.double().to(self.device)
			eta_plus.requires_grad = False
			eta_minus.requires_grad = False
			omega.requires_grad = False
			self.optimizer.zero_grad()
			output = self.model(data).double()
			loss = self.train_criterion(output, eta_plus, eta_minus, omega)
			loss.backward()
			self.optimizer.step()
			if batch_idx % log_interval == 0:
				print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
						   100. * batch_idx / len(train_loader), loss.item()))
			total_loss += loss.item()
		return total_loss

	# todo: consider if this is necessary
	def _test(self, test_loader):
		self.model.eval()
		test_loss = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(self.device), target.double().to(self.device)
				output = self.model(data).double().view(-1, )
				test_loss += self.test_criterion(output, target).item()
		test_loss /= len(test_loader.dataset)
		print('Test set: Average loss: {:.8f}\n'.format(test_loss))

	def predict(self, val_X):
		val_X = pd_to_numpy(val_X)
		self.model.eval()
		with torch.no_grad():
			score = self.model(torch.from_numpy(val_X).float().to(self.device)).double().view(-1,).cpu().numpy()
		prediction = self.score_to_label(score)
		return prediction

	def get_accuracy(self, test_X, test_y, prediction=None):
		test_y = pd_to_numpy(test_y)
		if prediction is not None:
			return accuracy_score(test_y, prediction)
		test_X = pd_to_numpy(test_X)
		return accuracy_score(test_y, self.predict(test_X))

	def get_balanced_accuracy(self, test_X, test_y, prediction=None):
		test_y = pd_to_numpy(test_y)
		if prediction is not None:
			return balanced_accuracy_score(test_y, prediction)
		test_X = pd_to_numpy(test_X)
		return balanced_accuracy_score(test_y, self.predict(test_X))

	def get_nosiy_loss(self, val_X_raw, bag_id_raw, prop_dict, size_dict, seed=None):
		val_X = pd_to_numpy(val_X_raw)
		pairs, bag_to_pair = generate_pairs(set(bag_id_raw), prop_dict, seed=seed, strategy=self.pairing_strategy, verbose=False)
		val_X, bag_id, pairs, mask = drop_invalid_bags(val_X, bag_id_raw, pairs, bag_to_pair)

		val_output = self.model(torch.from_numpy(val_X).float().to(self.device))

		weights = assign_weights(pairs, strategy=self.weighting_strategy)
		pair_to_size = pair_size(size_dict, pairs)

		val_omega = compute_omega(np.vectorize(bag_to_pair.get)(bag_id), weights, pair_to_size)
		val_noisy_y = recover_noisy_labels(bag_id, bag_to_pair, pairs)
		val_eta_plus, val_eta_minus = compute_eta(np.vectorize(bag_to_pair.get)(bag_id), pairs, val_noisy_y)

		return self.train_criterion(val_output.double().to(self.device), torch.from_numpy(val_eta_plus).double().to(self.device),
									torch.from_numpy(val_eta_minus).double().to(self.device), torch.from_numpy(val_omega).double().to(self.device)).item()

	def get_bag_loss(self, pred, bag_id, prop_dict):
		pred = pd.DataFrame(data=pred, index=bag_id)
		pred_prop, _ = compute_label_proportion(pred, bag_id, label=1)
		bag_error = 0
		for id, prop in pred_prop.items():
			bag_error += np.abs(prop_dict[id] - prop)
		return bag_error



'''
#check the gradient
from torch.autograd import gradcheck
target = torch.randn(50, 1, dtype=torch.double)
target = torch.where(target > 0, torch.ones(target.size()), -torch.ones(target.size())).double()
target.requires_grad = False
eta_plus = torch.exp(torch.randn(50, 1, dtype=torch.double, requires_grad=False)) - 1
eta_minus = torch.exp(torch.randn(50, 1, dtype=torch.double, requires_grad=False)) - 2
omega = torch.exp(torch.randn(50, 1, dtype=torch.double, requires_grad=False)) - 3
input = (torch.randn(50, 1, dtype=torch.double, requires_grad=True),
		 eta_plus,
		 eta_minus,
		 omega
         )
test_res = gradcheck(correct_loss(LogisticLossHelper).apply, input, eps=1e-7, atol=1e-4)
print(test_res)
'''

