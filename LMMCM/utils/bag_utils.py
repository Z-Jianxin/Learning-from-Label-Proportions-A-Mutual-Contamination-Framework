import pandas as pd
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import uniform


def assign_bags(strategy='random_n_size', random_seed=None, **kwargs):
	# Arguments:
	#     X: feature matrix, each feature vector should be represented as a row vector in the matrix
	#     num_bags: number of bags to make;
	#               will not effect the output if strategy==feature
	#     strategy: 'random': uniformly random with varying bag size, need arguments 'num_bags' and 'X'
	# 				'random_n_size': uniformly random with fixed bag size, need arguments 'num_bags' and 'X'
	# 				'feature': bag id is assigned based on the feature class, need arguments 'strategy_col' and 'X'
	# 				'multi-source': multi-source corruption i.e. given number of different bag proportions;
	# 								need arguments 'distribution', 'y', 'pos_label';
	#  									'y' is the label vector
	# 									'distribution' is a dictionary mapping (pos_instances, neg_instances) to the
	# 													number of bag under this distribution
	# 				'uniform_prop': for each bag, first generate a proportion with respect to a distribution,
	# 								then generate the labels w.r.t Bernoulli distribution；
	# 								need argument 'distribution', 'X', 'y', 'size', and 'pos_label';
	# 									'X' is the feature matrix
	# 									'y' is the label vector
	# 									'distribution' is a dictionary mapping [left_end, right_end] to the
	# 	 												number of bag with this distribution
	# 									'bag_size' is the size of a bag
	#     strategy_col: if strategy is 'feature', strategy_col is the pandas Series of that column
	#     random_seed:
	#
	# Functionality:
	#     assign bag id each instance; will NOT modify X
	#
	# Returns:
	# 	  (if the strategy is 'uniform_prop', returns X, y, bag_id)
	#     bag_id: a numpy ndarray of bag ids, corresponding to X by location;
	# 				bag ids are integers from 0 to X.shape[0]

	if random_seed is not None:
		np.random.seed(random_seed)  # fix a random seed if given

	# assign random bag index to instances, bag size can vary
	if strategy == 'random':
		num_bags = kwargs['num_bags']
		X = kwargs['X']
		bag_id = np.random.randint(0, high=num_bags, size=X.shape[0])

	# assign random bag index to instances, bag size is fixed
	elif strategy == 'random_n_size':
		num_bags = kwargs['num_bags']
		X = kwargs['X']

		# check if the number of instances is divisible by the number of bags
		assert X.shape[0] % num_bags == 0, \
			"number of instances %d is not divisible by number of bags %d" % (X.shape[0], num_bags)

		n = X.shape[0] // num_bags  # compute the size of each bag

		# assign bag index by appending integers to a 1d DataFrame and shuffling it.
		bag_id = pd.DataFrame(0, index=range(n), columns=['bag_id'])
		for i in range(1, num_bags):
			temp = pd.DataFrame(i, index=range(n), columns=['bag_id'])
			bag_id = bag_id.append(temp, ignore_index=True)
		np.random.shuffle(bag_id.values)
		bag_id = bag_id.values.reshape(-1, )

	# this is the method used in "no label no cry" code
	elif strategy == 'feature':
		strategy_col = kwargs['strategy_col']
		X = kwargs['X']
		bag_id = pd.Categorical(X[strategy_col]).codes

	# assign bag ids with desired label proportions
	elif strategy == 'multisource':
		distr = kwargs['distribution']
		y = kwargs['y']
		pos_label = kwargs['pos_label']

		bag_id = _multisource_helper(distr, y, pos_label)

	elif strategy == 'uniform_prop':
		distr = kwargs['distribution']
		X = kwargs['X']
		y = kwargs['y']
		pos_label = kwargs['pos_label']
		bag_size = kwargs['bag_size']

		distr_ = {}  # dictionary mapping (pos_instances, neg_instances) to the number of bag under
		for interval, num in distr.items():
			left, right = interval
			for i in range(num):
				prob = uniform.rvs(loc=left, scale=right - left)
				pos_num = bernoulli.rvs(prob, size=bag_size).sum()
				neg_num = bag_size - pos_num
				if not ((pos_num, neg_num) in distr_.keys()):
					distr_[(pos_num, neg_num)] = 0
				distr_[(pos_num, neg_num)] += 1

		pos_total = (y == pos_label).astype(int).sum()
		neg_total = (y != pos_label).astype(int).sum()
		pos_in_bag = 0
		neg_in_bag = 0
		for prop, num in distr_.items():
			pos_in_bag += prop[0] * num
			neg_in_bag += prop[1] * num

		# check the number of labels
		assert pos_in_bag <= pos_total, "insufficient positive labels, expect %d, have %d" % (pos_in_bag, pos_total)
		assert neg_in_bag <= neg_total, "insufficient negative labels, expect %d, have %d" % (neg_in_bag, neg_total)
		# done checking

		# sample labels
		X_pos = X[y == pos_label]
		y_pos = y[y == pos_label]
		X_neg = X[y != pos_label]
		y_neg = y[y != pos_label]

		random_perm_pos = np.random.permutation(X_pos.index)
		X_pos_shuffled = X_pos.reindex(random_perm_pos)
		y_pos_shuffled = y_pos.reindex(random_perm_pos)
		X_pos_sample = X_pos_shuffled[:pos_in_bag]
		y_pos_sample = y_pos_shuffled[:pos_in_bag]

		random_perm_neg = np.random.permutation(X_neg.index)
		X_neg_shuffled = X_neg.reindex(random_perm_neg)
		y_neg_shuffled = y_neg.reindex(random_perm_neg)
		X_neg_sample = X_neg_shuffled[:neg_in_bag]
		y_neg_sample = y_neg_shuffled[:neg_in_bag]

		new_X = pd.concat([X_pos_sample, X_neg_sample], ignore_index=True)
		new_y = pd.concat([y_pos_sample, y_neg_sample], ignore_index=True)

		return new_X, new_y, _multisource_helper(distr_, new_y, pos_label)

	else:
		raise NameError("unknown bag strategy: " + strategy)

	return bag_id


def _multisource_helper(distr, y, pos_label):
	# Arguments:
	#  	  'y' is the label vector
	# 	  'distribution' is a dictionary mapping (pos_instances, neg_instances) to the number of bag under
	# 	    	this distribution
	# Functionality:
	# 	  assign bag ids with the given distribution
	# Returns:
	# 	  bag_id

	# first need to check the number of pos/neg instances
	pos_num = (y == pos_label).astype(int).sum()
	neg_num = (y != pos_label).astype(int).sum()
	pos_in_bag = 0
	neg_in_bag = 0
	for prop, num in distr.items():
		pos_in_bag += prop[0] * num
		neg_in_bag += prop[1] * num
	assert pos_num == pos_in_bag
	assert neg_num == neg_in_bag
	# done checking

	num_bags = sum(distr.values())
	bag_id_pos = None
	bag_id_neg = None
	id = 0
	for prop, num in distr.items():
		for i in range(num):
			temp_pos = pd.DataFrame(id, index=range(prop[0]), columns=['bag_id'])
			temp_neg = pd.DataFrame(id, index=range(prop[1]), columns=['bag_id'])
			if bag_id_pos is None:
				bag_id_pos = temp_pos
			else:
				bag_id_pos = bag_id_pos.append(temp_pos, ignore_index=True)
			if bag_id_neg is None:
				bag_id_neg = temp_neg
			else:
				bag_id_neg = bag_id_neg.append(temp_neg, ignore_index=True)
			id += 1
	bag_id_pos = bag_id_pos.values  # bag_id_pos[i] is the bag id for the ith positive instance
	bag_id_neg = bag_id_neg.values  # bag_id_neg[i] is the bag id for the ith negative instance
	np.random.shuffle(bag_id_pos)
	np.random.shuffle(bag_id_neg)
	y = y.values
	bag_id = np.vectorize(
		lambda x: bag_id_pos[(y[:x] == y[x]).astype(int).sum()] if y[x] == pos_label else bag_id_neg[
			(y[:x] == y[x]).astype(int).sum()])(np.arange(y.shape[0]))
	return bag_id


def compute_label_proportion(y, bag_id, label=1):
	# Arguments:
	#     y: label vector.
	#     bag_id: a numpy ndarray of bag ids, corresponding to y by location
	#     label: the label value of which the proportion will be computed
	#
	# Functionality:
	#     compute the proportion of the given label in each bag
	#     compute the bag size
	#
	# Returns:
	#     prop_dict: a python dictionary mapping bag id to its proportion
	#     size_dict: a python dictionary mapping bag id to the bag size

	y_copy = y.copy()
	y_copy.index = bag_id
	prop = y_copy.groupby(y_copy.index).apply(lambda x: np.asscalar(((x == label).astype(float).sum()) / x.count()))
	size = y_copy.groupby(y_copy.index).apply(lambda x: np.asscalar(x.count()))
	prop_dict = {}
	size_dict = {}
	ids = prop.index.values
	for i in range(ids.shape[0]):
		prop_dict[ids[i]] = prop.loc[ids[i]]
		size_dict[ids[i]] = size.loc[ids[i]]
	return prop_dict, size_dict
