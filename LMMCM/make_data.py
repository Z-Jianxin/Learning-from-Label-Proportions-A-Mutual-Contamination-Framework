from utils.data_utils import *
from utils.bag_utils import *
from utils.model_utils import *
import pickle
import sys

import random
random.seed(2020)
np.random.seed(2020)

def load_adult():  # use 128*64=8192 instances for training, and 3000 for testing
	"""
	data info:
	training size: 32560; testing size: 16280
	training positive (>50k): 7840; training negative (<=50k): 24720
	testing positive (>50k): 3845; testing negative (<=50k): 12435
	label is +/- 1
	encoded dimension: 108
	"""

	train_path = "./experiments/data/adult/adult.data"
	test_path = "./experiments/data/adult/adult.test"
	label_name = 'income'
	csv_header = None
	manual_encode = {label_name: {"<=50K": -1, "<=50K.": -1, ">50K": 1, ">50K.": 1}}

	# optional params:
	adult_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
				   'marital-status', 'occupation', 'relationship', 'race',
				   'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
				   'native-country', 'income']
	cat_col = [adult_names[i] for i in [1, 3, 5, 6, 7, 8, 9, 13]]

	# load data
	train_data = csv_loader(train_path, csv_header, manual_encode, sep=', ', col_names=adult_names,
							drop_first_line=False, drop_last_line=True)
	test_data = csv_loader(test_path, csv_header, manual_encode, sep=', ', col_names=adult_names, drop_first_line=True,
						   drop_last_line=True)

	for col_name in cat_col:
		assert set(test_data[col_name].unique()) - set(train_data[col_name].unique()) == set({}), \
			"unmatched feature value, in training set not in testing set in \"%s\", %d, %d" % \
			(col_name, train_data[col_name].unique().size, test_data[col_name].unique().size)
	train_X, train_y = feature_label_split(train_data, label_name)
	test_X, test_y = feature_label_split(test_data, label_name)

	return train_X, train_y, test_X, test_y, cat_col


def check_info(loader):
	train_X, train_y, test_X, test_y, _ = loader()
	print("data info:")
	print("training size:", train_X.shape[0], "testing size:", test_X.shape[0])
	print("total positive:", (train_y == 1).sum() + (test_y == 1).sum(),
		  "total negative:", (train_y == -1).sum() + (test_y == -1).sum())
	print("training positive:", (train_y == 1).sum(), "training negative:", (train_y == -1).sum())
	print("testing positive:", (test_y == 1).sum(), "testing negative:", (test_y == -1).sum())
	print("label is +/- 1")
	print("encoded dimension:", train_X.shape[1])


def process_data(bag_size, loader, matpath, datapath, lp_distr, train_size=7680, test_size=3000, k_fold=5):
	"""
	example of matpath: "./experiments/mat/adult/%d/adult_BagSize_%d_Trial_%d.mat" % (bag_size, bag_size, trial)"
	example of datapath: "./experiments/py/adult/%d/adult_BagSize_%d_Trial_%d' % (bag_size, bag_size, trial)"
	"""
	assert train_size % (bag_size * 2) == 0
	train_X, train_y, test_X, test_y, cat_col = loader()
	train_X, test_X = feature_engineering_cat(train_X, test_X, cat_col)
	train_X = train_X.reset_index(drop=True)
	train_y = train_y.reset_index(drop=True)
	test_X = test_X.reset_index(drop=True)
	test_y = test_y.reset_index(drop=True)
	cont_col = list(set(train_X.columns.values) - set(cat_col))

	train_X, train_y, bag_id = assign_bags(strategy='uniform_prop',
											distribution={lp_distr: train_size//bag_size},  # 60 for size 128
											X=train_X, y=train_y, pos_label=1, bag_size=bag_size)

	prop_dict, size_dict = compute_label_proportion(train_y, bag_id, label=1)
	bag_to_fold = assign_fold(bag_id, k_fold)

	test_10_X, test_10_y = random_subset_by_lp(test_X, test_y, (test_size//10, 9 * test_size//10))  # 30, 270
	test_25_X, test_25_y = random_subset_by_lp(test_X, test_y, (test_size//4, 3 * test_size//4))    # 75, 225
	test_50_X, test_50_y = random_subset_by_lp(test_X, test_y, (test_size//2, test_size//2))		# 150, 150
	test_75_X, test_75_y = random_subset_by_lp(test_X, test_y, (3 * test_size//4, test_size//4))
	test_90_X, test_90_y = random_subset_by_lp(test_X, test_y, (9 * test_size//10, test_size//10))

	to_cv_matlab_pSVM(train_X, train_y, test_X, test_y, bag_id, prop_dict, cont_col, k_fold, bag_to_fold,
					  matpath,
					  extra_tests=True,
					  test_10_X=test_10_X, test_10_y=test_10_y,
					  test_25_X=test_25_X, test_25_y=test_25_y,
					  test_50_X=test_50_X, test_50_y=test_50_y,
					  test_75_X=test_75_X, test_75_y=test_75_y,
					  test_90_X=test_90_X, test_90_y=test_90_y)

	data_dict = dict()
	data_dict['train_X'] = train_X
	data_dict['train_y'] = train_y
	data_dict['test_X'] = test_X
	data_dict['test_y'] = test_y
	data_dict['bag_id'] = bag_id
	data_dict['prop_dict'] = prop_dict
	data_dict['size_dict'] = size_dict
	data_dict['k_fold'] = k_fold
	data_dict['cont_col'] = cont_col
	data_dict['bag_to_fold'] = bag_to_fold

	extra_tests = dict()
	extra_tests['test_10'] = (test_10_X, test_10_y)
	extra_tests['test_25'] = (test_25_X, test_25_y)
	extra_tests['test_50'] = (test_50_X, test_50_y)
	extra_tests['test_75'] = (test_75_X, test_75_y)
	extra_tests['test_90'] = (test_90_X, test_90_y)

	data_dict['extra_tests'] = extra_tests

	with open(datapath, 'wb') as data_file:
		pickle.dump(data_dict, data_file)


def load_shuttle():  # use 128*72 instances for training, and 3300 for testing
	"""
	data info:
	training size: 43500; testing size: 14500
	training positive (==1): 34108; training negative (!=1): 9392
	testing positive (==1): 11478; testing negative (!=1): 3022
	label is +/- 1
	encoded dimension: 9
	"""
	train_path = "./experiments/data/shuttle/shuttle.trn"
	test_path = "./experiments/data/shuttle/shuttle.tst"
	label_name = 9
	csv_header = None

	col_names = [j for j in range(10)]
	cat_col = []

	manual_encode = {}

	# load data
	train_data = csv_loader(train_path, csv_header, manual_encode, sep=' ', col_names=col_names,
							drop_first_line=False, drop_last_line=False)
	test_data = csv_loader(test_path, csv_header, manual_encode, sep=' ', col_names=col_names, drop_first_line=False,
						   drop_last_line=False)

	for col_name in cat_col:
		assert set(test_data[col_name].unique()) - set(train_data[col_name].unique()) == set({}), \
			"unmatched feature value, in training set not in testing set in \"%s\", %d, %d" % \
			(col_name, train_data[col_name].unique().size, test_data[col_name].unique().size)
	train_X, train_y = feature_label_split(train_data, label_name, label_map=lambda x: 1 if x == 1 else -1)
	test_X, test_y = feature_label_split(test_data, label_name, label_map=lambda x: 1 if x == 1 else -1)

	return train_X, train_y, test_X, test_y, cat_col


def load_dota():  # use 128*342 instances for training, and 10000 for testing
	"""
	data info:
	training size: 92650; testing size: 10294
	training positive: 48782; training negative: 43868
	testing positive: 5502; testing negative: 4792
	label is +/- 1
	"""
	train_path = "./experiments/data/dota2/dota2Train.csv"
	test_path = "./experiments/data/dota2/dota2Test.csv"
	label_name = '0'
	csv_header = None

	col_names = [str(j) for j in range(117)]
	cat_col = [str(j) for j in range(1, 117)]

	manual_encode = {}

	# load data
	train_data = csv_loader(train_path, csv_header, manual_encode, sep=',', col_names=col_names,
							drop_first_line=False, drop_last_line=False)
	test_data = csv_loader(test_path, csv_header, manual_encode, sep=',', col_names=col_names, drop_first_line=False,
						   drop_last_line=False)

	for col_name in cat_col:
		assert set(test_data[col_name].unique()) - set(train_data[col_name].unique()) == set({}), \
			"unmatched feature value, in training set not in testing set in \"%s\", %d, %d" % \
			(col_name, train_data[col_name].unique().size, test_data[col_name].unique().size)
	train_X, train_y = feature_label_split(train_data, label_name)
	test_X, test_y = feature_label_split(test_data, label_name)

	return train_X, train_y, test_X, test_y, cat_col


def load_magic():  # use 128*48=6144 instances for training, and 1400 for testing
	'''
	data info:
	training size: 15216; testing size: 3804; sample 20% for testing
	total positive: 12332; total negative: 6688
	estimated training positive: 9896; estimated training negative: 5320
	estimated testing positive: 2436; estimated testing negative: 1368
	label is +/- 1
	encoded dimension: 10
	'''
	data_path = "./experiments/data/magic/magic04.data"
	label_name = 'class'
	csv_header = None

	col_names = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist',
				 'class']
	cat_col = []

	manual_encode = {label_name: {'g': 1, 'h': -1}}

	# load data
	data = csv_loader(data_path, csv_header, manual_encode, sep=',', col_names=col_names,
						drop_first_line=False, drop_last_line=False)
	X, y = feature_label_split(data, label_name)
	train_X, train_y, test_X, test_y = train_test_split(X, y, int(X.shape[0] * 0.8), seed=None)
	train_X, test_X = feature_engineering_cat(train_X, test_X, cat_col)

	for col_name in cat_col:
		assert set(test_X[col_name].unique()) - set(train_X[col_name].unique()) == set({}), \
			"unmatched feature value, in training set not in testing set in \"%s\", %d, %d" % \
			(col_name, train_X[col_name].unique().size, test_X[col_name].unique().size)

	return train_X, train_y, test_X, test_y, cat_col


if __name__ == '__main__':
	"""
	usage: 
		python make_data.py loader dataset_name Matlab_data_folder Python_data_folder lower upper train_size test_size [number of bags]
	examples: 
		python make_data.py load_shuttle shuttle_test_make_data ./experiments/test_make_data/ ./experiments/test_make_data/ 0.5 1 5120 1000
		python make_data.py load_shuttle shuttle_test_make_data ./experiments/test_make_data/ ./experiments/test_make_data/ 0.5 1 -1 1000 10
	"""
	loader = globals()[sys.argv[1]]
	dataset_name = sys.argv[2]
	mat_folder = sys.argv[3]
	data_folder = sys.argv[4]
	lower = float(sys.argv[5])
	upper = float(sys.argv[6])
	train_size = int(sys.argv[7])
	test_size = int(sys.argv[8])
	is_fixed_num_bags = train_size
	for bag_size in [8, 16, 32, 64, 128, 256, 512]:
		if is_fixed_num_bags == -1:  # use fixed number of bags and vary bag sizes
			num_bags = int(sys.argv[9])
			train_size = num_bags * bag_size
		for i in range(5):
			process_data(bag_size=bag_size, loader=loader,
						 matpath=mat_folder + dataset_name + ("_%d_%d.mat" % (bag_size, i)),
						 datapath=data_folder + dataset_name + ("_%d_%d" % (bag_size, i)),
						 lp_distr=(lower, upper),
						 train_size=train_size, test_size=test_size, k_fold=5
						 )
