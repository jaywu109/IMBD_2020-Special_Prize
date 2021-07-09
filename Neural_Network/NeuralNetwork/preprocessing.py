# Modules
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd

target = 'Predict'

# Skewness
def skewness(df, input_features, random_state=None):
	pd.options.mode.chained_assignment = None 
	y = df[target]  # Get Y
	df = df[input_features] # Select columns

	# Skewness
	for feature in input_features:
		if abs(df[feature].skew()) >= 0.7:
			pt = PowerTransformer()
			d = pt.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
			d.shape = (1, len(d))
			df.loc[:, [feature]] = d.transpose()

	X = df.values.tolist()
	scaler = RobustScaler()
	df[df.columns] = scaler.fit_transform(X)

	X = df.values.tolist()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)  # Split train/test
	return X_train, X_test, y_train.tolist(), y_test.tolist()

# Min-max Normalization
def min_max_normalization(df, input_features, random_state=None):
	pd.options.mode.chained_assignment = None 
	y = df[target]  # Get Y
	df = df[input_features] # Select columns

	# Normalization
	for feature in df.columns:
		min_val = df[feature].min()
		max_val = df[feature].max()
		df[feature] = (df[feature] - min_val) / (max_val - min_val) if (max_val - min_val) else 0

	X = df.values.tolist()  # Get X
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)  # Split train/test

	return X_train, X_test, y_train.tolist(), y_test.tolist()

# Skewness
def skewness_pure(df, input_features, random_state=None):
	pd.options.mode.chained_assignment = None
	df = df[input_features] # Select columns
	
	# Skewness
	for feature in input_features:
		if abs(df[feature].skew()) >= 0.7:
			pt = PowerTransformer()
			d = pt.fit_transform(df[feature].values.reshape(-1, 1)).flatten()
			d.shape = (1, len(d))
			df.loc[:, [feature]] = d.transpose()

	X = df.values.tolist()
	scaler = RobustScaler()
	df[df.columns] = scaler.fit_transform(X)

	X = df.values.tolist()
	return X

# Min-max Normalization
def min_max_normalization_pure(df, input_features, random_state=None):
	pd.options.mode.chained_assignment = None 
	df = df[input_features] # Select columns

	# Normalization
	for feature in df.columns:
		min_val = df[feature].min()
		max_val = df[feature].max()
		df[feature] = (df[feature] - min_val) / (max_val - min_val) if (max_val - min_val) else 0

	X = df.values.tolist()  # Get X
	return X
