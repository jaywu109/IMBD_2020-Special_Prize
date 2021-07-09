from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers, metrics

# Input_A1_020
def Input_A1_020(weights_dir):
	input_features = ['Input_A4_020', 'Input_A3_020']
	model = models.load_model(weights_dir + 'Input_A1_020.h5')
	return model, input_features

# Input_A2_016
def Input_A2_016(weights_dir):
	input_features = ['Input_A2_019', 'Input_A2_018', 'Input_A4_016', 'Input_A6_017', 'Input_A6_016', 'Input_A5_019']
	model = models.load_model(weights_dir + 'Input_A2_016.h5')
	return model, input_features

# Input_A2_017
def Input_A2_017(weights_dir):
	input_features = ['Input_A2_018', 'Input_A2_019', 'Input_A1_019', 'Input_A5_016']
	model = models.load_model(weights_dir + 'Input_A2_017.h5')
	return model, input_features

# Input_A2_024
def Input_A2_024(weights_dir):
	input_features = ['Input_A2_022', 'Input_A2_023', 'Input_A3_024', 'Input_A3_022']
	model = models.load_model(weights_dir + 'Input_A2_024.h5')
	return model, input_features

# Input_A3_013
def Input_A3_013(weights_dir):
	input_features = ['Input_A1_013', 'Input_A6_014', 'Input_A1_014', 'Input_A2_013', 'Input_A6_013']
	model = models.load_model(weights_dir + 'Input_A3_013.h5')
	return model, input_features

# Input_A3_015
def Input_A3_015(weights_dir):
	input_features = ['Input_A2_015', 'Input_A6_015', 'Input_A1_015', 'Input_A4_015', 'Input_A5_015']
	model = models.load_model(weights_dir + 'Input_A3_015.h5')
	return model, input_features

# Input_A3_016
def Input_A3_016(weights_dir):
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A6_016', 'Input_A1_016']
	model = models.load_model(weights_dir + 'Input_A3_016.h5')
	return model, input_features

# Input_A3_017
def Input_A3_017(weights_dir):
	input_features = ['Input_A3_019', 'Input_A1_016', 'Input_A4_016', 'Input_A2_019']
	model = models.load_model(weights_dir + 'Input_A3_017.h5')
	return model, input_features

# Input_A3_018
def Input_A3_018(weights_dir):
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A5_019', 'Input_A1_018']
	model = models.load_model(weights_dir + 'Input_A3_018.h5')
	return model, input_features

# Input_A6_001
def Input_A6_001(weights_dir):
	input_features = ['Input_A6_002', 'Input_A6_003']
	model = models.load_model(weights_dir + 'Input_A6_001.h5')
	return model, input_features

# Input_A6_011
def Input_A6_011(weights_dir):
	input_features = ['Input_A5_011', 'Input_A2_011', 'Input_A1_011', 'Input_A3_011', 'Input_A4_011']
	model = models.load_model(weights_dir + 'Input_A6_011.h5')
	return model, input_features

# Input_A6_019
def Input_A6_019(weights_dir):
	input_features = ['Input_A3_019', 'Input_A2_019', 'Input_A6_016', 'Input_A1_016']
	model = models.load_model(weights_dir + 'Input_A6_019.h5')
	return model, input_features

# Input_A6_024
def Input_A6_024(weights_dir):
	input_features = ['Input_A2_022', 'Input_A2_023', 'Input_A3_024', 'Input_A3_022', 'Input_A3_023']
	model = models.load_model(weights_dir + 'Input_A6_024.h5')
	return model, input_features

# Input_C_013
def Input_C_013(weights_dir):
	input_features = ['Input_A6_016', 'Input_A6_018', 'Input_A2_019', 'Input_A5_016', 'Input_A5_018', 'Input_A4_019', 'Input_A5_017', 'Input_A4_016', 'Input_A5_019']
	model = models.load_model(weights_dir + 'Input_C_013.h5')
	return model, input_features

# Input_C_046
def Input_C_046(weights_dir):
	input_features = ['Input_C_043', 'Input_C_044', 'Input_C_041', 'Input_C_045']
	model = models.load_model(weights_dir + 'Input_C_046.h5')
	return model, input_features

# Input_C_049
def Input_C_049(weights_dir):
	input_features = ['Input_C_047', 'Input_C_048', 'Input_C_137']
	model = models.load_model(weights_dir + 'Input_C_049.h5')
	return model, input_features

# Input_C_050
def Input_C_050(weights_dir):
	input_features = ["Input_C_135", "Input_A1_011", "Input_A3_011", "Input_C_059", "Input_C_056", "Input_A2_011", "Input_C_051", "Input_A4_011", "Input_A5_011", "Input_C_053", "Input_C_055"]
	model = models.load_model(weights_dir + 'Input_C_050.h5')
	return model, input_features

# Input_C_057
def Input_C_057(weights_dir):
	input_features = ['Input_C_055', 'Input_C_135', 'Input_C_056', 'Input_C_052']
	model = models.load_model(weights_dir + 'Input_C_057.h5')
	return model, input_features

# Input_C_058
def Input_C_058(weights_dir):
	input_features = ['Input_C_054', 'Input_C_052', 'Input_C_055', 'Input_C_135']
	model = models.load_model(weights_dir + 'Input_C_058.h5')
	return model, input_features

# Input_C_096
def Input_C_096(weights_dir):
	input_features = ["Input_C_094", "Input_C_099", "Input_C_098", "Input_A2_014", "Input_C_044"]
	model = models.load_model(weights_dir + 'Input_C_096.h5')
	return model, input_features