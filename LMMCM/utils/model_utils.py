import numpy as np
import pandas as pd
from utils.data_utils import feature_engineering_cont, pd_to_numpy
from scipy.io import savemat
from scipy.special import expit

import time
import itertools
from copy import deepcopy

from pathos.helpers import mp
from pathos.multiprocessing import Pool
import multiprocessing

import networkx as nx
from networkx.algorithms.matching import max_weight_matching, is_perfect_matching, is_maximal_matching

from tabulate import tabulate


class LogisticLoss:
	def __init__(self):
		#self.forward = lambda t, y: np.log(1 + np.exp(-y * t))
		self.forward = lambda t, y: np.log(np.divide(1, expit(y * t)))
		#self.backward = lambda t, y: np.divide(-y, 1 + np.exp(y * t))
		self.backward = lambda t, y: -y * expit(-y * t)
		self.second_derivative = lambda t, y: np.divide(np.square(y) * np.exp(y * t), np.square(expit(y * t)))
		self.predict = lambda t: (t > 0).astype(int) * 2 - 1


class SigmoidLoss:
	def __init__(self):
		# self.forward = lambda t, y: np.divide(1.0, 1 + np.exp(y * t))
		self.forward = lambda t, y: expit(-y * t)
		self.backward = lambda t, y: (-y) * self.forward(t, y) * (1 - self.forward(t, y))
		self.second_derivative = lambda t, y: np.square(y) * self.forward(t, y) * (1 - self.forward(t, y)) \
											  * (1 - 2 * self.forward(t, y))
		self.predict = lambda t: (t > 0).astype(int) * 2 - 1


class Cross_Validation:

	def __init__(self, model, X, y, bag_id, prop_dict, size_dict, k_fold, cont_col, bag_to_fold=None, **kwargs):
		assert isinstance(X, pd.DataFrame), "the type of X should be pandas Dataframe, but it's actually %s"%type(X)
		assert isinstance(y, pd.core.series.Series), \
			"the type of y should be pandas Series, but it's actually %s"%type(y)
		assert (isinstance(bag_id, np.ndarray))
		self.model = model  										# model should have methods:
																	# fit, predict, get_accuracy, get_balanced_accuracy,
																	# get_nosiy_loss

		self.X = X  												# feature matrix, this should be unencoded raw data
		self.y = y  												# labels
		self.bag_id = bag_id
		self.prop_dict = prop_dict
		self.size_dict = size_dict
		self.kwargs = kwargs  										# kwargs that will be used in model.fit
		self.k = k_fold  											# the number of folds
		if bag_to_fold is None:
			self.bag_to_fold = assign_fold(self.bag_id, self.k)
		else:
			self.bag_to_fold = bag_to_fold
		self.cont_col = cont_col

		self.cv_models = dict()

	def _leave_fold_i(self, i):
		# leave the i-th fold
		# returns train_X, train_y, val_X, val_y, train_bag_ids, val_bag_ids
		train_mask = np.vectorize(lambda x: self.bag_to_fold[x] != i)(self.bag_id)
		validation_mask = ~train_mask
		return self.X[train_mask], self.y[train_mask], self.X[validation_mask], self.y[validation_mask], \
			   self.bag_id[train_mask], self.bag_id[validation_mask]

	def _cv_helper(self, i):
		self.cv_models[i] = deepcopy(self.model)
		train_X, train_y, val_X, val_y, train_bags, test_bags = self._leave_fold_i(i)
		train_X, val_X = feature_engineering_cont(train_X, val_X, self.cont_col)
		self.cv_models[i].fit(train_X, val_X, val_y, train_bags, self.prop_dict, self.size_dict, test_bags=test_bags,
							  **self.kwargs)
		pred = self.cv_models[i].predict(val_X)
		res_acc_i = self.cv_models[i].get_accuracy(val_X, val_y, pred)
		res_bacc_i = self.cv_models[i].get_balanced_accuracy(val_X, val_y, pred)
		res_noisy_loss_i = self.cv_models[i].get_test_obj(val_X, test_bags, self.prop_dict, self.size_dict)
		res_bag_loss_i = self.cv_models[i].get_bag_loss(pred, test_bags, self.prop_dict)
		return res_acc_i, res_bacc_i, res_noisy_loss_i, res_bag_loss_i

	def perform_cv(self):
		res_acc = np.empty((self.k,))
		res_bacc = np.empty((self.k,))
		res_noisy_loss = np.empty((self.k,))
		res_bag_loss = np.empty((self.k,))
		res_acc[:] = np.NaN
		res_bacc[:] = np.NaN
		res_noisy_loss[:] = np.NaN
		res_bag_loss[:] = np.NaN
		# with Pool(self.k) as p:
		#	self.res_list = p.map(self._cv_helper, range(self.k))
		#for i in range(self.k):
		#	res_acc[i], res_bacc[i], res_noisy_loss[i], res_bag_loss[i] = self.res_list[i]
		for i in range(self.k):
			res_acc[i], res_bacc[i], res_noisy_loss[i], res_bag_loss[i] = self._cv_helper(i)
		return (res_acc, res_bacc, res_noisy_loss, res_bag_loss), self.cv_models

	def get_model(self):
		return self.cv_models


def assign_fold(bag_id, k):
	# Arguments:
	#     bag_id: a numpy ndarray of bag ids, corresponding to feature matrix by location
	# 	  k: the number of folds
	#
	# Functionality:
	# 	  assign the a fold id to each bag; used in cross validation
	#
	# Returns:
	#     bag_to_fold: a python dictionary mapping a bag id to a fold id

	assert (len(set(bag_id)) >= 2*k), "don't have enough pairs for %d-fold cross validation"
	size = 2 * ((len(set(bag_id)) // 2) // k)
	reminder = (len(set(bag_id)) // 2) % k
	fold_id = np.full((size, ), 0)
	for i in range(1, k):
		if i < k - reminder:
			fold_id = np.append(fold_id, np.full((size, ), i))
		else:
			fold_id = np.append(fold_id, np.full((size + 2, ), i))
	fold_id = np.random.permutation(fold_id)
	assert (fold_id.shape[0] == len(set(bag_id)))
	bag_to_fold = dict(zip(list(set(bag_id)), fold_id))
	return bag_to_fold


def to_cv_matlab_pSVM(train_X_raw, train_y, test_X_raw, test_y, bag_id, prop_dict, cont_col, k_fold, bag_to_fold,
					  filename, prop_label=1, extra_tests=False, **kwargs):
	# Todo: implement this to convert data into matlab data on which we could run the pSVM experiments
	# data: X
	# split:
	# 		train_data_idx: 1 * 206
	# 		train_bag_idx: 206 * 1
	# 		train_bag_prop: [, ]
	# 		train_label: 206 * 1
	# 		test_data_idx: 1 * 64
	# 		test_bag_idx: 64 * 1
	# 		test_bag_prop:
	# 		test_label: 64 * 1
	# 		inner_split: 1*5, this seems to be useless
	# Todo: consider to avoid the recomputation of kernel
	res = to_data_split_matlab(train_X_raw, train_y, test_X_raw, test_y, bag_id, prop_dict, cont_col,
							   prop_label=prop_label, extra_tests = extra_tests, **kwargs)

	folds = dict()
	res['folds'] = folds
	folds['k_fold'] = k_fold
	for i in range(k_fold):
		train_mask = np.vectorize(lambda x: bag_to_fold[x] != i)(bag_id)
		validation_mask = ~train_mask
		train_X_fold, train_y_fold, val_X, val_y, bag_id_fold = train_X_raw[train_mask], train_y[train_mask], \
																train_X_raw[validation_mask], \
																train_y[validation_mask], bag_id[train_mask]
		bag_id_fold_val = bag_id[validation_mask]
		folds['fold_%d'%i] = to_data_split_matlab(train_X_fold, train_y_fold, val_X, val_y, bag_id_fold, prop_dict,
												  cont_col, bag_id_test=bag_id_fold_val)
	savemat(filename, res)


def to_data_split_matlab(train_X_raw, train_y, test_X_raw, test_y, bag_id, prop_dict, cont_col, bag_id_test=None,
						 prop_label=1, extra_tests=False, **kwargs):
	# helper function to convert the data for MATLAB code
	train_X, test_X = feature_engineering_cont(train_X_raw, test_X_raw, cont_col)

	train_X = pd_to_numpy(train_X)
	train_y = pd_to_numpy(train_y)
	test_X = pd_to_numpy(test_X)
	test_y = pd_to_numpy(test_y)

	X = np.concatenate((train_X, test_X), axis=0)
	train_data_idx = np.arange(1, train_X.shape[0] + 1)
	test_data_idx = np.arange(train_X.shape[0] + 1, X.shape[0] + 1)

	# create a new set of indices
	new_idx_to_bag_id = dict()
	bag_id_to_new_idx = dict()
	i = 1
	for old_bag in set(bag_id):
		new_idx_to_bag_id[i] = old_bag
		bag_id_to_new_idx[old_bag] = i
		i += 1

	train_bag_idx = np.vectorize(bag_id_to_new_idx.get)(bag_id)
	train_bag_prop = np.vectorize(lambda x: prop_dict[new_idx_to_bag_id[x]])(np.arange(1, len(set(bag_id)) + 1))

	# Todo: pack the information of test bags
	test_bag_idx = np.ones((test_X.shape[0],))
	test_bag_prop = (test_y == prop_label).mean()
	if bag_id_test is not None:
		new_idx_to_bag_id_test = dict()
		bag_id_to_new_idx_test = dict()
		i = 1
		for old_bag in set(bag_id_test):
			new_idx_to_bag_id_test[i] = old_bag
			bag_id_to_new_idx_test[old_bag] = i
			i += 1
		test_bag_idx = np.vectorize(bag_id_to_new_idx_test.get)(bag_id_test)
		test_bag_prop = np.vectorize(lambda x: prop_dict[new_idx_to_bag_id_test[x]])(np.arange(1, len(set(bag_id_test)) + 1))

	res = dict()
	split = dict()

	# for the new testing scheme
	if extra_tests:
		for i in [10, 25, 50, 75, 90]:
			_, test_X_extra = feature_engineering_cont(train_X_raw, kwargs['test_%d_X' % i], cont_col)
			test_X_extra = pd_to_numpy(test_X_extra)
			test_y_extra = pd_to_numpy(kwargs['test_%d_y' % i])
			split['test_%d_data_idx' % i] = np.arange(X.shape[0] + 1, X.shape[0] + kwargs['test_%d_X' % i].shape[0] + 1)
			X = np.concatenate((X, test_X_extra), axis=0)
			split['test_%d_bag_idx' % i] = np.ones((test_X_extra.shape[0],))
			split['test_%d_bag_prop' % i] = (test_y_extra == prop_label).mean()
			split['test_%d_label' % i] = (test_y_extra * 1.0).reshape(-1, 1)

			split['test_%d_data_idx' % i] = (split['test_%d_data_idx' % i] * 1.0).reshape(1, -1)
			split['test_%d_bag_idx' % i] = (split['test_%d_bag_idx' % i] * 1.0).reshape(-1, 1)
			split['test_%d_bag_prop' % i] = (split['test_%d_bag_prop' % i] * 1.0).reshape(-1, 1)

	#

	res['data'] = X
	res['split'] = split
	split['train_data_idx'] = (train_data_idx * 1.0).reshape(1, -1)
	split['train_bag_idx'] = (train_bag_idx * 1.0).reshape(-1, 1)
	split['train_bag_prop'] = (train_bag_prop * 1.0).reshape(-1, 1)
	split['train_label'] = (train_y * 1.0).reshape(-1, 1)
	split['test_data_idx'] = (test_data_idx * 1.0).reshape(1, -1)
	split['test_bag_idx'] = (test_bag_idx * 1.0).reshape(-1, 1)
	split['test_bag_prop'] = (test_bag_prop * 1.0).reshape(-1, 1)
	split['test_label'] = (test_y * 1.0).reshape(-1, 1)

	return res


class GridSearch():

	def __init__(self, train_X, train_y, test_X, test_y, bag_id, prop_dict, size_dict, k_fold, cont_col,
				 bag_to_fold, model_params, train_params, MODEL):
		self.train_X = train_X  				# training feature matrix
		self.train_y = train_y  				# training labels
		self.test_X = test_X					# testing feature matrix
		self.test_y = test_y 					# testing labels
		self.bag_id = bag_id 					# numpy nd array of bag ids, corresponding to train_X by position
		self.prop_dict = prop_dict				# dictionary mapping bag id to its proportion
		self.size_dict = size_dict				# dictionary mapping bag id to its size
		self.k = k_fold							# the number of folds
		self.cont_col = cont_col					# the names of categorical columns
		self.bag_to_fold = bag_to_fold			# a python dictionary mapping a bag id to a fold id;
												# 	if ==None, will call assign_fold()
		self.model_params = model_params		# model parameters to search
		self.train_params = train_params		# training parameters to search

		self.kernel_param_names = [key for key in self.model_params.keys()]
		self.train_param_names = [train for train in self.train_params.keys()]
		self.kernel_param_lists = [self.model_params[key] for key in self.kernel_param_names] # the name kernel params is left unchanged
		self.train_param_lists = [self.train_params[key] for key in self.train_params]

		self.res_dict = dict()
		# self.model_dict = dict()  # store models in grid search
		self.best_kernel_params_dict = {'balanced_accuracy': None, 'accuracy': None, 'noisy_loss': None, 'bag_loss': None}
		self.best_train_params_dict = {'balanced_accuracy': None, 'accuracy': None, 'noisy_loss': None, 'bag_loss': None}
		self.final_res_dict = {}
		for key in ['acc', 'bacc', 'auc', 'cm', 'roc']:
			self.final_res_dict[key] = {'balanced_accuracy': None, 'accuracy': None, 'noisy_loss': None, 'bag_loss': None}
		self.final_models_dict = {'balanced_accuracy': None, 'accuracy': None, 'noisy_loss': None, 'bag_loss': None}

		self.extra_tests_res_dict = dict()

		self.MODEL = MODEL

	def _fill_grids(self):
		self.num_grids = sum(1 for _ in itertools.product(*self.kernel_param_lists)) * sum(
			1 for _ in itertools.product(*self.train_param_lists))
		self.param_dict_list = []
		self.param_tuple_list = []
		cv_list_temp = []

		for kernel_params_tuple in itertools.product(*self.kernel_param_lists):
			kernel_param_dict = {self.kernel_param_names[i]: kernel_params_tuple[i] for i in range(len(kernel_params_tuple))}
			for train_params_tuple in itertools.product(*self.train_param_lists):
				train_param_dict = {self.train_param_names[i]: train_params_tuple[i] for i in range(len(train_params_tuple))}
				model = self.MODEL(**kernel_param_dict)
				cv = Cross_Validation(model, self.train_X, self.train_y, self.bag_id, self.prop_dict, self.size_dict,
									  self.k, self.cont_col, bag_to_fold=self.bag_to_fold, **train_param_dict)
				self.param_dict_list.append((kernel_param_dict, train_param_dict))
				self.param_tuple_list.append((kernel_params_tuple, train_params_tuple))
				cv_list_temp.append(cv)

		with Pool(multiprocessing.cpu_count()) as p:
			temp = p.map(lambda cv_: cv_.perform_cv(), cv_list_temp)
		self.res_list = [ele[0] for ele in temp]
		self.cv_list = [ele[1] for ele in temp]

	def _search_grids(self):
		max_performance = {'balanced_accuracy': float("-inf"), 'accuracy': float("-inf"), 'noisy_loss': float("inf"),
						   'bag_loss': float("inf")}
		for i in range(len(self.param_dict_list)):
			kernel_param_dict, train_param_dict = self.param_dict_list[i]
			kernel_params_tuple, train_params_tuple = self.param_tuple_list[i]
			res_acc, res_bacc, noisy_loss, bag_loss = self.res_list[i]
			accuracy_mean = np.nanmean(res_acc)
			balanced_accuracy_mean = np.nanmean(res_bacc)
			noisy_loss_mean = np.nanmean(noisy_loss)
			bag_loss_mean = np.nanmean(bag_loss)
			self.res_dict[(kernel_params_tuple, train_params_tuple)] = ((accuracy_mean, np.nanstd(res_acc)),
																		(balanced_accuracy_mean, np.nanstd(res_bacc)),
																		(noisy_loss_mean, np.nanstd(noisy_loss)),
																		(bag_loss_mean, np.nanstd(bag_loss)))

			if accuracy_mean >= max_performance['accuracy']:
				max_performance['accuracy'] = accuracy_mean
				self.best_kernel_params_dict['accuracy'] = kernel_param_dict
				self.best_train_params_dict['accuracy'] = train_param_dict
			if balanced_accuracy_mean >= max_performance['balanced_accuracy']:
				max_performance['balanced_accuracy'] = balanced_accuracy_mean
				self.best_kernel_params_dict['balanced_accuracy'] = kernel_param_dict
				self.best_train_params_dict['balanced_accuracy'] = train_param_dict
			if noisy_loss_mean <= max_performance['noisy_loss']:
				max_performance['noisy_loss'] = noisy_loss_mean
				self.best_kernel_params_dict['noisy_loss'] = kernel_param_dict
				self.best_train_params_dict['noisy_loss'] = train_param_dict
			if bag_loss_mean <= max_performance['bag_loss']:
				max_performance['bag_loss'] = bag_loss_mean
				self.best_kernel_params_dict['bag_loss'] = kernel_param_dict
				self.best_train_params_dict['bag_loss'] = train_param_dict

	def perform_gs(self):
		self._fill_grids()
		self._search_grids()

	def print_search_results(self, outfile):
		file = open(outfile, "w+")
		for key in self.res_dict.keys():
			for i in range(len(self.kernel_param_names)):
				file.write("%s = %s, " % (self.kernel_param_names[i], key[0][i]))
			for i in range(len(self.train_param_names)):
				file.write("%s = %s, " % (self.train_param_names[i], key[1][i]))
			file.write("\nbag_loss_mean = %.4f, bag_loss_std = %.4f \n" % (self.res_dict[key][3][0], self.res_dict[key][3][1]))
			file.write("noisy_loss_mean = %.4f, noisy_loss_std = %.4f \n" % (self.res_dict[key][2][0], self.res_dict[key][2][1]))
			file.write("accuracy_mean = %.4f, accuracy_std = %.4f, balanced_accuracy_mean = %.4f, balanced_accuracy_std = %.4f, \n\n" %
					   (self.res_dict[key][0][0], self.res_dict[key][0][1], self.res_dict[key][1][0], self.res_dict[key][1][1]))
		file.write("\n\n\n")
		file.write("final train:\n")
		file.write("best params: { \n")
		for key, value in self.best_kernel_params_dict.items():
			file.write('%s : %s, \n' % (key, value))
			file.write('%s : %s, \n' % (key, self.best_train_params_dict[key]))
		file.write("\n}\n")
		for metric in self.final_res_dict['acc'].keys():
			file.write("\n\n")
			if self.final_res_dict['acc'][metric] is not None:
				file.write("%s :\n" % metric)
				file.write("accuracy = %.4f\n" % (self.final_res_dict['acc'][metric]))
			if self.final_res_dict['bacc'][metric] is not None:
				file.write("balanced_accuracy = %.4f\n" % (self.final_res_dict['bacc'][metric]))
			if self.final_res_dict['auc'][metric] is not None:
				file.write("auc = %.4f\n" % (self.final_res_dict['auc'][metric]))
			if self.final_res_dict['cm'][metric] is not None:
				cm = self.final_res_dict['cm'][metric]
				file.write("confusion_matrix =\n %s\n" % (tabulate([['true pos', cm[0, 0], cm[0, 1]],
																	['true neg', cm[1, 0], cm[1,1]]],
																	headers=['pred pos', 'pred neg'],
																	tablefmt="grid")))
			file.write("\n\n")

		file.write("\n\nextra tests results:")
		for metric in self.extra_tests_res_dict.keys():
			file.write("\n\n")
			file.write("%s: \n" % metric)
			for key, value in self.extra_tests_res_dict[metric].items():
				file.write("\n%s: acc=%.4f, bacc=%.4f" % (key, value[0], value[1]))
			file.write("\n\n")
		file.close()

	def final_train_test(self, metric, extra_tests=None):
		assert (self.best_kernel_params_dict[metric] is not None)
		assert (self.best_train_params_dict[metric] is not None)
		final_model = self.MODEL(**self.best_kernel_params_dict[metric])
		train_X, test_X = feature_engineering_cont(self.train_X, self.test_X, self.cont_col)
		final_model.fit(train_X, test_X, self.test_y, self.bag_id, self.prop_dict, self.size_dict,
						**self.best_train_params_dict[metric])
		pred = final_model.predict(test_X)
		self.final_res_dict['acc'][metric] = final_model.get_accuracy(test_X, self.test_y, prediction=pred)
		self.final_res_dict['bacc'][metric] = final_model.get_balanced_accuracy(test_X, self.test_y, prediction=pred)
		self.final_res_dict['cm'][metric] = final_model.get_confusion_matrix(test_X, self.test_y, prediction=pred)
		area, fprs, tprs, thresholds = final_model.get_roc(test_X, self.test_y)
		self.final_res_dict['auc'][metric] = area
		self.final_res_dict['roc'][metric] = (fprs, tprs, thresholds)
		self.final_models_dict[metric] = final_model

		if extra_tests is not None:
			self.extra_tests_res_dict[metric] = dict()
			for key in extra_tests.keys():
				test_X, test_y = extra_tests[key]
				_, test_X = feature_engineering_cont(self.train_X, test_X, self.cont_col)
				pred = final_model.predict(test_X)
				res = (final_model.get_accuracy(test_X, test_y, prediction=pred),
					   final_model.get_balanced_accuracy(test_X, test_y, prediction=pred))
				self.extra_tests_res_dict[metric][key] = res


def compute_omega(pair_ids, weights, pair_size_dict):
	# Arguments:
	#     pair_ids: a numpy ndarray of pair ids, corresponding to training instances by location
	#     weights: a python dictionary mapping a pair id to the associated weight
	#     pair_size_dict: a python dictionary mapping a pair id to its size
	#
	# Functionality:
	# 	  compute the weight assigned to each instance, i.e. the omega vector in the manual;
	# 	  internal use only
	#
	# Returns:
	#     omega

	instance_weight = np.vectorize(weights.get)(pair_ids)
	instance_weight = np.divide(instance_weight, np.vectorize(pair_size_dict.get)(pair_ids))
	return instance_weight


def compute_eta(pair_ids, pairs, noisy_y):
	# Arguments:
	#     pair_ids: a numpy ndarray of pair ids, corresponding to training instances by location
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	# 	  noisy_y: noisy labels
	#
	# Functionality:
	# 	  compute the eta_plus, eta_minus vectors explained in the manual;
	# 	  internal use only
	#
	# Returns:
	#     eta_plus, eta_minus

	gamma_plus = np.vectorize(lambda i: pairs[i][1][0])(pair_ids)
	gamma_minus = np.vectorize(lambda i: pairs[i][1][1])(pair_ids)
	diff = gamma_plus - gamma_minus
	# the label must be +/- 1
	eta_plus = np.divide(1 - gamma_minus, diff) * (noisy_y == 1) + np.divide(gamma_plus - 1, diff) * (noisy_y == -1)
	eta_minus = np.divide(-gamma_minus, diff) * (noisy_y == 1) + np.divide(gamma_plus, diff) * (noisy_y == -1)
	return eta_plus, eta_minus


def generate_pairs(ids, prop_dict, strategy='greedy', seed=None):
	# Arguments:
	#     ids: the set of bag ids; the number of bags must be even
	#     prop_dict: a dictionary mapping bag id to its label proportion
	#     strategy: the strategy of pairing;
	#               could be 'random', 'fixed_size_optimal'
	#
	# Functionality:
	#     make bag pairs for Corrected Loss Algorithm
	#
	# Returns:
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	#     bag_to_pair: a dictionary mapping a bag id to corresponding pair id

	num_bags = len(ids)
	num_pairs = num_bags // 2
	if seed is not None:
		np.random.seed(seed)
	assert num_bags % 2 == 0, "can't pair odd number of bags: %d" % (num_bags,)

	# assign pair ids to bags;
	# generate a numpy array called pair_id;
	# pair_id[i] is the pair id associated with ids[i]
	if strategy == 'random':
		ids = list(ids)
		pair_id = np.append(np.arange(num_pairs), np.arange(num_pairs))
		np.random.shuffle(pair_id)
	elif strategy == 'fixed_size_optimal':
		# produce the matrix in numpy 2d ndarray
		ids = list(ids)
		graph_matrix = np.zeros((num_bags, num_bags))
		for i in range(num_bags):
			for j in range(i + 1, num_bags):
				graph_matrix[i][j] = (prop_dict[ids[i]] - prop_dict[ids[j]]) ** 2 + 1  # +1 so the graph is complete
				# graph_matrix[i][j] = (prop_dict[i] - prop_dict[j]) ** 2 + 1
				graph_matrix[j][i] = graph_matrix[i][j]
		graph = nx.Graph(graph_matrix)
		start_time = time.time()
		matching = max_weight_matching(graph, maxcardinality=True)
		elapsed_time = time.time() - start_time
		pair_id = np.zeros((len(ids, )))
		# assert (is_maximal_matching(graph, matching)) # should run this, but apparently take too long
		# assert (is_perfect_matching(graph, matching)) # should run this, but apparently take too long
		assert (len(matching) == num_pairs)
		matching = list(matching)
		for idx in range(num_pairs):
			i, j = matching[idx]
			pair_id[i] = idx
			pair_id[j] = idx
	# pair_id[ids[i]] = idx
	# pair_id[ids[j]] = idx
	elif strategy == 'greedy':
		ids = list(ids)
		ids.sort(key=lambda x: prop_dict[x])
		pair_id = np.append(np.arange(num_pairs), np.arange(num_pairs - 1, -1, -1))
	else:
		print("Unknown pairing strategy", strategy)
		raise NameError("Unknown pairing strategy")

	pairs = {}
	bag_to_pair = {}
	ids = list(ids)
	for i in range(len(ids)):
		if pair_id[i] not in pairs.keys():
			pairs[pair_id[i]] = []
		pairs[pair_id[i]].append(ids[i])
	for key in pairs.keys():
		bag_plus, bag_minus = pairs[key][0], pairs[key][1]
		gamma_plus, gamma_minus = prop_dict[bag_plus], prop_dict[bag_minus]
		if gamma_plus < gamma_minus:
			bag_plus, bag_minus = bag_minus, bag_plus
			gamma_plus, gamma_minus = prop_dict[bag_plus], prop_dict[bag_minus]
		pairs[key] = ((bag_plus, bag_minus), (gamma_plus, gamma_minus))
		bag_to_pair[bag_plus] = key
		bag_to_pair[bag_minus] = key
	return pairs, bag_to_pair


def assign_weights(pairs, strategy='uniform'):
	# Arguments:
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	#     strategy: the bagging strategy;
	#               could be 'uniform' or 'optimal'
	#
	# Functionality:
	#     assign weights to each pair
	#
	# Returns:
	#     weights: a dictionary mapping the pair id to the associated weights;
	# 				the weights should sum up to 1

	weights = {}
	if strategy == 'uniform':
		weight = 1 / len(pairs)
		for pair_id in pairs:
			weights[pair_id] = weight
		return weights
	elif strategy == 'fixed_size_optimal':
		sum_weights = 0
		for pair_id in pairs.keys():
			gamma_0, gamma_1 = pairs[pair_id][1]
			weight = (gamma_0 - gamma_1) ** 2
			weights[pair_id] = weight
			sum_weights += weight
		for pair_id in weights.keys():
			weights[pair_id] /= sum_weights
	return weights


def pair_size(size_dict, pairs):
	# Arguments:
	# 	  size_dict: a python dictionary mapping bag id to the bag size
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	#
	# Functionality:
	#     compute the size of each pair
	#
	# Returns:
	# 	  pair_to_size: a python dictionary mapping pair id to the pair size
	pair_to_size = {}
	for pair_id in pairs.keys():
		bag0, bag1 = pairs[pair_id][0]
		pair_to_size[pair_id] = size_dict[bag0] + size_dict[bag1]
	return pair_to_size


def recover_noisy_labels(bag_id, bag_to_pair, pairs):
	# Arguments:
	#     bag_id: a numpy ndarray of bag ids, corresponding to X by location;
	# 				bag ids are integers from 0 to X.shape[0]
	# 	  bag_to_pair: a dictionary mapping a bag id to corresponding pair id
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	#
	# Functionality:
	# 	  map the instances in the positive bag of each pair to +1 and those in negative bags to -1
	#
	# Returns:
	#     noisy_y: a numpy array of +/- 1's, which are the noisy labels of bags,
	# 				corresponding to bag_id by position

	noisy_y = np.vectorize(lambda x: x == pairs[bag_to_pair[x]][0][0])(bag_id)
	noisy_y = noisy_y.astype(int) * 2 - 1
	return noisy_y


def drop_invalid_bags(train_X, bag_id, pairs, bag_to_pair):
	# Arguments:
	# 	  train_X: the training feature matrix
	#     bag_id: a numpy ndarray of bag ids, corresponding to X by location;
	# 				bag ids are integers from 0 to X.shape[0]
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	# 	  bag_to_pair: a dictionary mapping a bag id to corresponding pair id
	#
	# Functionality:
	# 	  drop the invalid bag pairs from train_X, bag_id, and pairs
	#
	# Returns:
	# 	  the training feature matrix after dropping the invalid bag paris;
	# 	 	 original orders are preserved
	# 	  a list of valid bag ids
	# 	  pairs: same structure as the input argument pair; the invalid pairs are dropped
	# 	  mask: a binary numpy ndarray;
	# 	 		 1 signifies an instance in a valid bag;
	# 			 corresponding to train_X by position

	# valid bag pairs should satisfy gamma+ != gamma-
	valid_bags = set([bag for bag in bag_id if pairs[bag_to_pair[bag]][1][0] != pairs[bag_to_pair[bag]][1][1]])
	mask = np.vectorize(lambda x: bag_id[x] in valid_bags)(np.arange(train_X.shape[0]))
	pairs = dict((pair_id, pairs[pair_id]) for pair_id in pairs.keys() if pairs[pair_id][0][0] in valid_bags)
	if len(valid_bags) == 0:
		print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("NO BAGS AFTER DROPPING INVALID")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
		raise ValueError('All bags are dropped')
	return train_X[mask, :], bag_id[mask], pairs, mask


def exclude_by_heuristic(train_X, bag_id, pairs, bag_to_pair, k = 2):
	# Arguments:
	# 	  train_X: the training feature matrix
	#     bag_id: a numpy ndarray of bag ids, corresponding to X by location;
	# 				bag ids are integers from 0 to X.shape[0]
	#     pairs: a dictionary mapping a pair id to a tuple ((B+, B-), (gamma+, gamma-));
	# 				B+ is the id of the positive bag and B- is the id of the negative bag;
	# 				gamma+ and gamma- are the relative bag proportions;
	# 				it's asserted that gamma+ >= gamma-;
	# 	  bag_to_pair: a dictionary mapping a bag id to corresponding pair id
	# 	  k: the confidence parameter
	#
	# Functionality:
	#     drop the bag pairs by the exclusion heuristic
	#
	# Returns:
	# 	  the training feature matrix after dropping the invalid bag paris;
	# 	 	 original orders are preserved
	# 	  a list of valid bag ids
	# 	  pairs: same structure as the input argument pair; the invalid pairs are dropped
	# 	  mask: a binary numpy ndarray;
	# 	 		 1 signifies an instance in a valid bag;
	# 			 corresponding to train_X by position
	gamma_plus = {bag: pairs[bag_to_pair[bag]][1][0] for bag in bag_id}
	gamma_minus = {bag: pairs[bag_to_pair[bag]][1][1] for bag in bag_id}
	num_pairs = len(set(bag_id)) / 2
	valid_bags = set([bag for bag in bag_id if (gamma_plus[bag] - gamma_minus[bag]) >=
					  k * np.sqrt((gamma_plus[bag] + gamma_minus[bag] - gamma_plus[bag]**2 -
								   gamma_minus[bag]**2)/num_pairs)])
	mask = np.vectorize(lambda x: bag_id[x] in valid_bags)(np.arange(train_X.shape[0]))
	pairs = dict((pair_id, pairs[pair_id]) for pair_id in pairs.keys() if pairs[pair_id][0][0] in valid_bags)
	if len(valid_bags) == 0:
		print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("k = %d, ALL BAGS ARE EXCLUDED" % k)
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
		raise ValueError('All bags are excluded')
	return train_X[mask, :], bag_id[mask], pairs, mask
