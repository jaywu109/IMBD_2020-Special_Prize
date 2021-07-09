from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics

# Input_A1_020
def Input_A1_020():
	input_features = ['Input_A4_020', 'Input_A3_020']
	model = models.Sequential()
	model.add(layers.Dense(12, activation='relu',input_shape=(2,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A2_016
def Input_A2_016():
	input_features = ['Input_A2_019', 'Input_A2_018', 'Input_A4_016', 'Input_A6_017', 'Input_A6_016', 'Input_A5_019']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='relu',input_shape=(6,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A2_017
def Input_A2_017():
	input_features = ['Input_A2_018', 'Input_A2_019', 'Input_A1_019', 'Input_A5_016']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='sigmoid',input_shape=(4,)))
	model.add(layers.Dense(14, activation='sigmoid'))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A2_024
def Input_A2_024():
	input_features = ['Input_A2_022', 'Input_A2_023', 'Input_A3_024', 'Input_A3_022']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='relu',input_shape=(4,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A3_013
def Input_A3_013():
	input_features = ['Input_A1_013', 'Input_A6_014', 'Input_A1_014', 'Input_A2_013', 'Input_A6_013']
	model = models.Sequential()
	model.add(layers.Dense(6, activation='relu',input_shape=(5,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A3_015
def Input_A3_015():
	input_features = ['Input_A2_015', 'Input_A6_015', 'Input_A1_015', 'Input_A4_015', 'Input_A5_015']
	model = models.Sequential()
	model.add(layers.Dense(25, activation='relu',input_shape=(5,)))
	model.add(layers.Dropout(0.1))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A3_016
def Input_A3_016():
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A6_016', 'Input_A1_016']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='sigmoid',input_shape=(4,)))
	model.add(layers.Dense(14, activation='sigmoid'))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A3_017
def Input_A3_017():
	input_features = ['Input_A3_019', 'Input_A1_016', 'Input_A4_016', 'Input_A2_019']
	model = models.Sequential()
	model.add(layers.Dense(20, activation='sigmoid',input_shape=(4,)))
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(5, activation='relu'))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A3_018
def Input_A3_018():
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A5_019', 'Input_A1_018']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='sigmoid',input_shape=(4,)))
	model.add(layers.Dense(14, activation='sigmoid'))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A6_001
def Input_A6_001():
	input_features = ['Input_A6_002', 'Input_A6_003']
	model = models.Sequential()
	model.add(layers.Dense(22, activation='relu',input_shape=(2,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A6_011
def Input_A6_011():
	input_features = ['Input_A5_011', 'Input_A2_011', 'Input_A1_011', 'Input_A3_011', 'Input_A4_011']
	model = models.Sequential()
	model.add(layers.Dense(4, activation='relu',input_shape=(5,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A6_019
def Input_A6_019():
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A6_016', 'Input_A1_016']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='sigmoid',input_shape=(4,)))
	model.add(layers.Dense(14, activation='sigmoid'))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_A6_024
def Input_A6_024():
	input_features = ['Input_A2_022', 'Input_A2_023', 'Input_A3_024', 'Input_A3_022', 'Input_A3_023']
	model = models.Sequential()
	model.add(layers.Dense(4, activation='relu',input_shape=(5,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_013
def Input_C_013():
	input_features = ['Input_A6_016', 'Input_A6_018', 'Input_A2_019', 'Input_A5_016', 'Input_A5_018', 'Input_A4_019', 'Input_A5_017', 'Input_A4_016', 'Input_A5_019']
	model = models.Sequential()
	model.add(layers.Dense(2, activation='sigmoid',input_shape=(9,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_046
def Input_C_046():
	input_features = ['Input_C_043', 'Input_C_044', 'Input_C_041', 'Input_C_045']
	model = models.Sequential()
	model.add(layers.Dense(8, activation='relu',input_shape=(4,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_049
def Input_C_049():
	input_features = ['Input_C_047', 'Input_C_048', 'Input_C_137']
	model = models.Sequential()
	model.add(layers.Dense(5, activation='relu',input_shape=(3,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_050
def Input_C_050():
	input_features = ["Input_C_135", "Input_A1_011", "Input_A3_011", "Input_C_059", "Input_C_056", "Input_A2_011", "Input_C_051", "Input_A4_011", "Input_A5_011", "Input_C_053", "Input_C_055"]
	model = models.Sequential()
	model.add(layers.Dense(6, activation='relu',input_shape=(11,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_057
def Input_C_057():
	input_features = ['Input_C_055', 'Input_C_135', 'Input_C_056', 'Input_C_052']
	model = models.Sequential()
	model.add(layers.Dense(7, activation='relu',input_shape=(4,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_058
def Input_C_058():
	input_features = ['Input_C_054', 'Input_C_052', 'Input_C_055', 'Input_C_135']
	model = models.Sequential()
	model.add(layers.Dense(6, activation='relu',input_shape=(4,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features

# Input_C_096
def Input_C_096():
	input_features = ["Input_C_094", "Input_C_099", "Input_C_098", "Input_A2_014", "Input_C_044"]
	model = models.Sequential()
	model.add(layers.Dense(6, activation='relu',input_shape=(5,)))
	model.add(layers.Dense(1, activation='linear'))
	model.compile(optimizer='adam', loss=metrics.mean_squared_error, metrics=[metrics.RootMeanSquaredError(name='rmse')])
	return model, input_features