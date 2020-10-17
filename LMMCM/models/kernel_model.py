from utils.model_utils import recover_noisy_labels, compute_omega, compute_eta, generate_pairs, drop_invalid_bags, \
	assign_weights, pair_size, exclude_by_heuristic
from utils.bag_utils import *
from utils.data_utils import *
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve, auc, confusion_matrix

from scipy.optimize import minimize


def _objective_helper(beta, K, omega, eta, loss, regularizer):
	eta_plus, eta_minus = eta
	t = K @ beta
	if regularizer != 0:
		train_obj = omega @ (eta_plus * loss.forward(t, np.ones((t.shape[0],))) +
							 eta_minus * loss.forward(t, -np.ones(t.shape[0], ))) + regularizer * beta @ K @ beta
	else:
		train_obj = omega @ (eta_plus * loss.forward(t, np.ones((t.shape[0],))) +
							 eta_minus * loss.forward(t, -np.ones(t.shape[0], )))
	return train_obj


def _objective_grad_helper(beta, K, omega, eta, loss, regularizer):
	eta_plus, eta_minus = eta
	t = K @ beta
	epsilon_plus = loss.backward(t, np.ones((t.shape[0],)))
	epsilon_minus = loss.backward(t, -np.ones(t.shape[0], ))
	grad = K @ (omega * eta_plus * epsilon_plus + omega * eta_minus * epsilon_minus) + 2 * regularizer * K @ beta
	return grad


def _objective_hessian_helper(beta, K, omega, eta, loss, regularizer):
	eta_plus, eta_minus = eta
	t = K @ beta
	epsilon_2_plus = loss.second_derivative(t, np.ones((t.shape[0],)))
	epsilon_2_minus = loss.second_derivative(t, -np.ones(t.shape[0], ))
	hessian = K @ np.diag(omega * (eta_plus * epsilon_2_plus + eta_minus * epsilon_2_minus)) @ K + 2 * regularizer * K
	return hessian


def objective(beta, K, loss, bag_id, pairs, bag_to_pair, pair_ids, pair_size_dict, weights, regularizer):
	noisy_y = recover_noisy_labels(bag_id, bag_to_pair, pairs)
	omega = compute_omega(pair_ids, weights, pair_size_dict)
	eta = compute_eta(pair_ids, pairs, noisy_y)
	return _objective_helper(beta, K, omega, eta, loss, regularizer)


def objective_grad(beta, K, loss, bag_id, pairs, bag_to_pair, pair_ids, pair_size_dict, weights, regularizer):
	noisy_y = recover_noisy_labels(bag_id, bag_to_pair, pairs)
	omega = compute_omega(pair_ids, weights, pair_size_dict)
	eta = compute_eta(pair_ids, pairs, noisy_y)
	return _objective_grad_helper(beta, K, omega, eta, loss, regularizer)


def objective_hessian(beta, K, loss, bag_id, pairs, bag_to_pair, pair_ids, pair_size_dict, weights, regularizer):
	noisy_y = recover_noisy_labels(bag_id, bag_to_pair, pairs)
	omega = compute_omega(pair_ids, weights, pair_size_dict)
	eta = compute_eta(pair_ids, pairs, noisy_y)
	return _objective_hessian_helper(beta, K, omega, eta, loss, regularizer)


class KernelizedMethod:

	def __init__(self, kernel, loss, regularizer, pairing_strategy='greedy',  weighting_strategy='fixed_size_optimal',
				 **kwargs):
		self.kernel = kernel
		self.kernel_params = kwargs
		self.loss = loss()
		self.is_trained = False
		self.regularizer = regularizer

		# methods to generate pairs and assign weights
		self.pairing_strategy = pairing_strategy
		self.weighting_strategy = weighting_strategy

		# lists to keep track of objectives
		self.acc_list = []
		self.bacc_list = []
		self.test_obj_list = []
		self.bag_loss_list = []
		self.train_obj_list = []

		# the model
		self.beta = None

	def fit(self, train_X, test_X, test_y, bag_id, prop_dict, size_dict, method, test_bags=None, seed=None,
			reinitialize=True, exclusion_param=0):
		train_X = pd_to_numpy(train_X)
		test_X = pd_to_numpy(test_X)
		test_y = pd_to_numpy(test_y)

		self.train_X = train_X

		# pair the bags
		self.pairs, self.bag_to_pair = generate_pairs(set(bag_id), prop_dict, seed=seed, strategy=self.pairing_strategy)
		# the function drop_invalid_bags raises critical exception and print debug messages
		try:
			self.train_X, self.bag_id, self.pairs, mask = drop_invalid_bags(train_X, bag_id, self.pairs,
																			self.bag_to_pair)
			self.train_X, self.bag_id, self.pairs, mask = exclude_by_heuristic(self.train_X, self.bag_id, self.pairs,
																		  self.bag_to_pair, k=exclusion_param)
		except ValueError:
			self.beta = None
			return

		self.weights = assign_weights(self.pairs, strategy=self.weighting_strategy)
		self.pair_size_dict = pair_size(size_dict, self.pairs)

		# compute the kernel matrix
		if self.kernel_params['gamma'] == 'scale':
			self.kernel_params['gamma'] = 1 / (self.train_X.shape[1] * self.train_X.var())
		elif self.kernel_params['gamma'] == 'auto':
			self.kernel_params['gamma'] = 1 / self.train_X.shape[1]
		self.K = pairwise_kernels(self.train_X, self.train_X, metric=self.kernel, **self.kernel_params)

		if (self.beta is None) or reinitialize:
			self.beta = np.random.uniform(low=-0.5, high=0.5, size=(self.train_X.shape[0],))  # initialize beta
		assert self.beta.shape[0] == self.train_X.shape[0]

		# optimization
		if method[0] == "vanilla_gradient_descent":
			max_iterations, lr, decay, stop_criterion = method[1]['max_iterations'], method[1]['lr'], \
														method[1]['decay'], method[1]['stop_criterion']
			self._vanilla_gradient_descent(max_iterations, lr, decay, stop_criterion, test_X, test_y, prop_dict,
										   size_dict, test_bags)
		else:
			if (method[0] == 'L-BFGS-B') or (method[0] == 'BFGS'):
				hess = None					# disable the warning
			else:
				hess = objective_hessian
			params = (self.K, self.loss, self.bag_id, self.pairs, self.bag_to_pair,
					  np.vectorize(self.bag_to_pair.get)(self.bag_id), self.pair_size_dict, self.weights,
					  self.regularizer)
			res = minimize(objective, self.beta, args=params, method=method[0], jac=objective_grad, hess=hess,
						   options=method[1])
			self.beta = res.x

	def _get_scores(self, test_X):
		test_X = pd_to_numpy(test_X)
		K_ = pairwise_kernels(test_X, self.train_X, metric=self.kernel, **self.kernel_params)
		return K_ @ self.beta

	def predict(self, test_X):
		if self.beta is None:
			return None
		return self.loss.predict(self._get_scores(test_X))

	def get_accuracy(self, test_X, test_y, prediction=None):
		if self.beta is None:
			return float("-inf")
		test_y = pd_to_numpy(test_y)
		if prediction is not None:
			return accuracy_score(test_y, prediction)
		test_X = pd_to_numpy(test_X)
		return accuracy_score(test_y, self.predict(test_X))

	def get_balanced_accuracy(self, test_X, test_y, prediction=None):
		if self.beta is None:
			return float("-inf")
		test_y = pd_to_numpy(test_y)
		if prediction is not None:
			return balanced_accuracy_score(test_y, prediction)
		test_X = pd_to_numpy(test_X)
		return balanced_accuracy_score(test_y, self.predict(test_X))

	def get_test_obj(self, test_X_raw, bag_id_raw, prop_dict, size_dict, seed=None):
		if self.beta is None:
			return float("inf")
		test_X = pd_to_numpy(test_X_raw)
		pairs, bag_to_pair = generate_pairs(set(bag_id_raw), prop_dict, seed=seed, strategy=self.pairing_strategy)
		try:
			test_X, bag_id, pairs, mask = drop_invalid_bags(test_X, bag_id_raw, pairs, bag_to_pair)
		except ValueError:
			print("EXCEPTION IN COMPUTING TESTING OBJECTIVE")
			return np.nan
		test_K = pairwise_kernels(test_X, self.train_X, metric=self.kernel, **self.kernel_params)
		weights = assign_weights(pairs, strategy=self.weighting_strategy)
		pair_size_dict = pair_size(size_dict, pairs)
		test_obj = objective(self.beta, test_K, self.loss, bag_id, pairs, bag_to_pair,
							 np.vectorize(bag_to_pair.get)(bag_id), pair_size_dict, weights, 0)
		return test_obj

	def get_bag_loss(self, pred, bag_id, prop_dict):
		if self.beta is None:
			return float("inf")
		pred = pd.DataFrame(data=pred, index=bag_id)
		pred_prop, _ = compute_label_proportion(pred, bag_id, label=1)
		bag_error = 0
		for id, prop in pred_prop.items():
			bag_error += np.abs(prop_dict[id] - prop)
		return bag_error

	def get_confusion_matrix(self, test_X, test_y, prediction=None):
		if self.beta is None:
			return None
		test_y = pd_to_numpy(test_y)
		if prediction is not None:
			return confusion_matrix(test_y, prediction, normalize='true')
		test_X = pd_to_numpy(test_X)
		return confusion_matrix(test_y, self.predict(test_X), normalize='true')

	def get_roc(self, test_X, test_y):
		if self.beta is None:
			return None, None, None, None
		test_X = pd_to_numpy(test_X)
		test_y = pd_to_numpy(test_y)
		scores = self._get_scores(test_X)
		fprs, tprs, thresholds = roc_curve(test_y, scores, pos_label=1)
		area = auc(fprs, tprs)
		return area, fprs, tprs, thresholds

	def _vanilla_gradient_descent(self, max_iterations, lr, decay, stop_criterion, test_X, test_y, prop_dict, size_dict,
								  test_bags):
		# optimization
		prev_train_obj = objective(self.beta, self.K, self.loss, self.bag_id, self.pairs, self.bag_to_pair,
								   np.vectorize(self.bag_to_pair.get)(self.bag_id), self.pair_size_dict,
								   self.weights, self.regularizer)
		for i in range(max_iterations):
			grad = objective_grad(self.beta, self.K, self.loss, self.bag_id, self.pairs, self.bag_to_pair,
								  np.vectorize(self.bag_to_pair.get)(self.bag_id), self.pair_size_dict, self.weights,
								  self.regularizer)
			self.beta -= lr * grad

			current_train_obj = objective(self.beta, self.K, self.loss, self.bag_id, self.pairs, self.bag_to_pair,
								  np.vectorize(self.bag_to_pair.get)(self.bag_id), self.pair_size_dict, self.weights,
								  self.regularizer)

			if abs(prev_train_obj - current_train_obj) < stop_criterion:
				break
			prev_train_obj = current_train_obj
			pred = self.predict(test_X)
			self.train_obj_list.append(current_train_obj)									# train objective
			test_acc = self.get_accuracy(test_X, test_y, prediction=pred)
			self.acc_list.append(test_acc)													# test/val accuracy
			test_bacc = self.get_balanced_accuracy(test_X, test_y, prediction=pred)
			self.bacc_list.append(test_bacc)												# test/val balanced acc
			if test_bags is not None:
				test_obj = self.get_test_obj(test_X, test_bags, prop_dict, size_dict)
				self.test_obj_list.append(test_obj)											# test objective
				bag_loss = self.get_bag_loss(pred, test_bags, prop_dict)
				self.bag_loss_list.append(bag_loss)											# bag loss
			lr *= np.exp(- decay)
