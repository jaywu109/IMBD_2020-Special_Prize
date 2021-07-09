# Modules
import pandas as pd
import NeuralNetwork.pretrained_models as pre_models
from NeuralNetwork.tools import predict, predict_mul
from NeuralNetwork.preprocessing import skewness_pure, min_max_normalization_pure


'''
# Data loading
df = pd.read_csv('Data/BayesianRidge_Pre_1/Input_A1_020.csv')
model, input_features = pre_models.Input_A1_020()

# Extract needed input feature
df = df[input_features].values.tolist()

# Predict single
p = predict(model, df[0])
print('Single prediction of Input_A1_020:')
print('X:')
print(df[0])
print('Y:')
print(p)

# Predict mutiple
p = predict_mul(model, df)
print('Multiple prediction of Input_A1_020:')
print('X:')
print(df)
print('Y:')
print(p)
'''

def predict_df(X_df):
	model_A1_020, input_features_A1_020 = pre_models.Input_A1_020('weights/')
	model_A2_016, input_features_A2_016 = pre_models.Input_A2_016('weights/')
	model_A2_017, input_features_A2_017 = pre_models.Input_A2_017('weights/')
	model_A2_024, input_features_A2_024 = pre_models.Input_A2_024('weights/')
	model_A3_013, input_features_A3_013 = pre_models.Input_A3_013('weights/')
	model_A3_015, input_features_A3_015 = pre_models.Input_A3_015('weights/')
	model_A3_016, input_features_A3_016 = pre_models.Input_A3_016('weights/')
	model_A3_017, input_features_A3_017 = pre_models.Input_A3_017('weights/')
	model_A3_018, input_features_A3_018 = pre_models.Input_A3_018('weights/')
	model_A6_001, input_features_A6_001 = pre_models.Input_A6_001('weights/')
	model_A6_011, input_features_A6_011 = pre_models.Input_A6_011('weights/')
	model_A6_019, input_features_A6_019 = pre_models.Input_A6_019('weights/')
	model_A6_024, input_features_A6_024 = pre_models.Input_A6_024('weights/')
	model_C_013, input_features_C_013 = pre_models.Input_C_013('weights/')
	model_C_046, input_features_C_046 = pre_models.Input_C_046('weights/')
	model_C_049, input_features_C_049 = pre_models.Input_C_049('weights/')
	model_C_050, input_features_C_050 = pre_models.Input_C_050('weights/')
	model_C_057, input_features_C_057 = pre_models.Input_C_057('weights/')
	model_C_058, input_features_C_058 = pre_models.Input_C_058('weights/')
	model_C_096, input_features_C_096 = pre_models.Input_C_096('weights/')

	outputs = [
		'Input_A1_020', 'Input_A2_016', 'Input_A2_017', 'Input_A2_024', 'Input_A3_013',
		'Input_A3_015', 'Input_A3_016', 'Input_A3_017', 'Input_A3_018', 'Input_A6_001',
		'Input_A6_011', 'Input_A6_019', 'Input_A6_024', 'Input_C_013', 'Input_C_046', 
		'Input_C_049', 'Input_C_050', 'Input_C_057', 'Input_C_058', 'Input_C_096'
	]

	preprocessings = [
		'skewness', 'skewness', 'min_max_normalization', 'skewness', 'skewness',
		'min_max_normalization', 'min_max_normalization', 'min_max_normalization', 'min_max_normalization', 'skewness',
		'min_max_normalization', 'min_max_normalization', 'min_max_normalization', 'min_max_normalization', 'min_max_normalization',
		'min_max_normalization', 'skewness', 'min_max_normalization', 'min_max_normalization', 'skewness',
		]

	models = [
		model_A1_020, model_A2_016, model_A2_017, model_A2_024, model_A3_013,
		model_A3_015, model_A3_016, model_A3_017, model_A3_018, model_A6_001,
		model_A6_011, model_A6_019, model_A6_024, model_C_013, model_C_046, 
		model_C_049, model_C_050, model_C_057, model_C_058, model_C_096
	]

	input_features = [
		input_features_A1_020, input_features_A2_016, input_features_A2_017, input_features_A2_024, input_features_A3_013,
		input_features_A3_015, input_features_A3_016, input_features_A3_017, input_features_A3_018, input_features_A6_001,
		input_features_A6_011, input_features_A6_019, input_features_A6_024, input_features_C_013, input_features_C_046, 
		input_features_C_049, input_features_C_050, input_features_C_057, input_features_C_058, input_features_C_096
	]

	result = {}
	for i in range(len(outputs)):

		if preprocessings[i] == 'skewness':
			X = skewness_pure(X_df, input_features[i])
		elif preprocessings[i] == 'min_max_normalization':
			X = min_max_normalization_pure(X_df, input_features[i])

		result[outputs[i]] = predict_mul(models[i], X)
    
	result_df = pd.DataFrame(result, columns = outputs)
	return result_df

df = pd.read_csv('../Y_20.csv')
df_result = predict_df(df)
print(df_result.head)