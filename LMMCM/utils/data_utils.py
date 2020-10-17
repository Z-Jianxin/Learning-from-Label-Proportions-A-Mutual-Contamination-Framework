import numpy as np
import pandas as pd
from sklearn import preprocessing
import types


def csv_loader(path, csv_header, manual_encoder, sep=',', col_names=None, drop_first_line=False, drop_last_line=False,
			   missing_handle=dict()):
	# Arguments:
	#     path: path to csv file;
	# 			used in function pandas.read_csv
	#     csv_header: 0 if the first row is the columns names; None if no header
	# 					used in function pandas.read_csv
	#     manual_encode: a dictionary of dictionaries (can be empty);
	#                    column_name->(original_value->encoded_value);
	#                    labels must be encoded manually
	#     sep: separators of csv
	# 			used in function pandas.read_csv
	#     col_names: optional, names of columns in data files
	# 					used in function pandas.read_csv
	#     drop_first_line: optional, binary, if ==True, drop the first row of the data file
	#     drop_last_line: optional, if ==True, drop the last row of the data file
	#     missing_handle: dictionary, specify the scheme to handle a CONTINUOUS missing value;
	#                     applied BEFORE the manual encoder;
	#                     column_name -> (missing symbol, handle scheme)
	# 						handle_scheme could be 'mean' to take the mean values
	#
	# Functionality:
	#     load a given csv file and encode its values;
	#     set the column names accordingly;
	#     solve the missing values;
	#     drop the last line of the file (might NaNs) if drop_last_line=True.
	#
	# Returns:
	#     df: the loaded dataframe.
	engine = 'c'
	if len(sep) > 1:
		engine = 'python'
	df = pd.read_csv(path, sep=sep, header=csv_header, names=col_names, engine=engine)
	if drop_first_line:
		df.drop(df.head(1).index, inplace=True)
	if drop_last_line:
		df.drop(df.tail(1).index, inplace=True)

	# handle missing values:
	for col in missing_handle.keys():
		symbol, scheme = missing_handle[col]
		if scheme == 'mean':
			df[col][df[col] == symbol] = df[col][df[col] != symbol].astype(float).mean()

	# Manual encoding
	for col in manual_encoder.keys():
		if col not in df.columns:
			print(col, "cannot be found in the data file:", path)
			raise NameError("header is not in the data file: " + path)
		if isinstance(manual_encoder[col], types.FunctionType):
			df[col] = df[col].apply(manual_encoder[col])
		elif isinstance(manual_encoder[col], dict):
			df[col] = df[col].map(manual_encoder[col])
		else:
			raise TypeError
	return df


def feature_label_split(df, label_col, label_map=None):
	# Arguments:
	#     df: DataFrame, containing features and labels.
	#     label_col: the column to be used as label; can be index if no name
	#     label_map:  original_value->encoded_value; could either be a function or a dictionary
	#
	# Functionality:
	#     split the input DataFrame of the form (X, y) into X, y;
	#     then have y encoded if needed.
	#
	# Returns:
	#     X: features.
	#     y: labels.

	if label_col not in df.columns:
		raise NameError("label cannot be found in the dataframe")
	y = df[label_col]
	X = df.drop(label_col, axis=1)
	if label_map is not None:
		if isinstance(label_map, types.FunctionType):
			y = y.apply(label_map)
		elif isinstance(label_map, dict):
			y = y.map(label_map)
	return X, y


def train_test_split(X, y, train_size, test_size=None, shuffle=True, seed=None):
	# Arguments:
	#     X: pandas DataFrame, features matrix
	#     y: labels
	#     train_size: size of training data
	#     test_size: optional; if None, then calculate based on train_size.
	#     shuffle: binary; if shuffle == True, then shuffle the X and y before slicing
	# 	  seed: random seed for shuffling; has no effect if shuffle is disabled
	#
	# Functionality:
	# 	  split a dataset into feature matrix and labels
	#
	# Returns:
	#     train_X
	#     train_y
	#     test_X
	#     test_y

	if shuffle:
		if seed is not None:
			np.random.seed(seed)
		random_perm = np.random.permutation(X.index)
		X = X.reindex(random_perm)
		y = y.reindex(random_perm)
	if test_size is None:
		return (X.iloc[:train_size], y.iloc[:train_size], X.iloc[train_size:], y.iloc[train_size:])
	assert (train_size + test_size <= X.shape[0])
	return (X.iloc[:train_size], y.iloc[:train_size], X.iloc[-test_size:], y.iloc[-test_size:])


def feature_engineering_cat(train_X, test_X, cat_col):
	# Arguments:
	#     train_X: training feature matrix
	# 	  test_X: testing feature matrix
	#     cat_col: the list of columns with categorical values
	#
	# Functionality:
	#     one hot encode all categorical values;
	#
	# Returns:
	#     train_X: the engineered training feature matrix
	#     test_X: the engineered testing feature matrix

	train_test = train_X.append(test_X, ignore_index=True)
	train_size = train_X.shape[0]
	# one-hot encode categorical values
	for i in np.intersect1d(cat_col, train_test.columns.values):
		temp = pd.get_dummies(train_test[i])
		names = [i + " %s" % j for j in range(temp.shape[1])]
		train_test[names] = temp
		del train_test[i]
	# Split train and test as before
	train_X_after = train_test[:train_size].copy()
	test_X_after = train_test[train_size:].copy()
	return train_X_after, test_X_after


def feature_engineering_cont(train_X, test_X, cont_col):
	# Arguments:
	#     train_X: training feature matrix
	# 	  test_X: testing feature matrix
	#     cat_col: the list of columns with continuous values
	#
	# Functionality:
	#     standardize all continuous values;
	#
	# Returns:
	#     train_X: the engineered training feature matrix
	#     test_X: the engineered testing feature matrix

	# standardize continuous features - also include dates
	train_X_after = train_X.copy()
	test_X_after = test_X.copy()
	if len(cont_col) != 0:
		scaler = preprocessing.StandardScaler()
		train_X_after.loc[:, cont_col] = scaler.fit_transform(train_X_after[cont_col])
		test_X_after.loc[:, cont_col] = scaler.transform(test_X_after[cont_col])
	return train_X_after, test_X_after

def random_subset(X, y, subset, seed=None):
	# Arguments:
	# 	  X: feature matrix
	# 	  y: labels
	# 	  subset: int, indicating size of subset
	# 	  seed: prefixed random seed
	#
	# Functionality:
	# 	  take a random subset of data
	#
	# Returns:
	# 	  X_subset
	# 	  y_subset
	if seed is not None:
		np.random.seed(seed)
	random_perm = np.random.permutation(X.index)
	X = X.reindex(random_perm)
	y = y.reindex(random_perm)
	return X[:subset], y[:subset]


def random_subset_by_lp(X, y, label_prop, seed=None, pos_label=1):
	# Arguments:
	# 	  X: feature matrix
	# 	  y: labels
	# 	  subset: int, indicating size of subset
	# 	  label_prop: tuple, (num_of_pos, num_of_neg)
	# 	  seed: prefixed random seed
	# 	  pos_label: the positive label
	#
	# Functionality:
	# 	  take a random subset of data with desired label proportions
	#
	# Returns:
	# 	  X_
	# 	  y_

	X_pos = X[y == pos_label]
	X_neg = X[y != pos_label]
	y_pos = y[y == pos_label]
	y_neg = y[y != pos_label]
	pos_num, neg_num = label_prop

	assert pos_num < X_pos.shape[0], "insufficient number of positive labels, will sample %d, have %d" % (pos_num,
																								   		X_pos.shape[0])
	assert neg_num < X_neg.shape[0], "insufficient number of negative labels, will sample %d, have %d" % (pos_num,
																								   		X_pos.shape[0])
	X_pos_sample, y_pos_sample = random_subset(X_pos, y_pos, pos_num, seed)
	X_neg_sample, y_neg_sample = random_subset(X_neg, y_neg, neg_num, seed)

	X_ = pd.concat([X_pos_sample, X_neg_sample], ignore_index=True)
	y_ = pd.concat([y_pos_sample, y_neg_sample], ignore_index=True)

	return X_, y_

def pd_to_numpy(data):
	# Arguments:
	# 	  data: a pandas DataFrame or Series
	#
	# Functionality:
	# 	  transform data into numpy an ndarray if it is a pandas DataFrame or Series;
	# 	  do nothing otherwise
	#
	# Returns:
	# 	  data: the embedded numpy ndarray of the input

	if isinstance(data, pd.DataFrame) or isinstance(data, pd.core.series.Series):
		data = data.values
	return data
