import time
from utils.model_utils import *
from models.kernel_model import *
import dill as pickle
import sys

import random
random.seed(2020)
np.random.seed(2020)

class hdict(dict):
	def __key(self):
		return tuple((k,self[k]) for k in sorted(self))
	def __hash__(self):
		return hash(self.__key())
	def __eq__(self, other):
		return self.__key() == other.__key()


def run_experiment(data_path, path_to_save_results, save_model=True, path_to_save_model=None):

	if save_model:
		assert path_to_save_model is not None

	print('experiment starts on %s' % data_path)
	experiment_start = time.time()

	# prepare data
	with open(data_path, 'rb') as data_file:
		data_dict = pickle.load(data_file)
	train_X = data_dict['train_X']
	train_y = data_dict['train_y']
	test_X = data_dict['test_X']
	test_y = data_dict['test_y']
	bag_id = data_dict['bag_id']
	prop_dict = data_dict['prop_dict']
	size_dict = data_dict['size_dict']
	k_fold = data_dict['k_fold']
	cont_col = data_dict['cont_col']
	bag_to_fold = data_dict['bag_to_fold']

	extra_tests = data_dict['extra_tests']

	# set up parameters
	KERNEL_PARAMS = {'kernel': ['rbf', ], 'gamma': ['scale', ], 'regularizer': [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
					 'loss': [LogisticLoss, ]}

	opt = ('L-BFGS-B', hdict({'ftol': 1e-5, 'maxiter': 100, 'maxcor': 80}))
	TRAIN_PARAMS = {'method': [opt, ], 'exclusion_param': [0, ]}
	# perform grid search
	gs = GridSearch(train_X, train_y, test_X, test_y, bag_id, prop_dict, size_dict, k_fold, cont_col,
					bag_to_fold, KERNEL_PARAMS, TRAIN_PARAMS, MODEL=KernelizedMethod)
	gs.perform_gs()
	gs.final_train_test(metric='bag_loss', extra_tests=extra_tests)
	gs.final_train_test(metric='noisy_loss', extra_tests=extra_tests)
	gs.print_search_results(path_to_save_results)

	if save_model:
		with open(path_to_save_model, 'wb') as data_file:
			pickle.dump({'gs': gs}, data_file)

	elapsed_time = time.time() - experiment_start
	print("Trial ends. Time elapsed: %.4fs\n" % elapsed_time)

if __name__ == '__main__':
	data_path = sys.argv[1]
	path_to_save_results = sys.argv[2]
	run_experiment(data_path, path_to_save_results, save_model=False)
	
