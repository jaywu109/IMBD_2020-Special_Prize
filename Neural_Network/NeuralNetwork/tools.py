#import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

def train(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=2000, save_dir='weights/temp.h5', show_plt=False, show_result=False):
	
	earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')

	# Train model
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	history = model.fit(
		X_train,
		y_train,
		epochs = epochs,
		validation_data = (X_val, y_val),
		batch_size = batch_size,
		shuffle = True,
		callbacks = [earlyStopping],
		verbose = 0
	)

	# Save weights
	if save_dir:
		model.save(save_dir)

	# Show plt
	if show_plt:
		def show_train_history(train_history):
			plt.plot(train_history.history['rmse'])
			plt.plot(train_history.history['val_rmse'])
			plt.xticks([i for i in range(0, len(train_history.history['rmse']))])
			plt.title('Train History')
			plt.ylabel('rmse')
			plt.xlabel('epoch')
			plt.legend(['train', 'validation'], loc='upper left')
			plt.show()

		show_train_history(history)

	if show_result:
		print('train:')
		for i in range(len(y_train)):
			print('ground truth: %f, predict: %f'%(y_train[i], model.predict([X_train[i]])[0][0] ))

		# Show validate result
		print('validate:')
		for i in range(len(y_val)):
			print('ground truth: %f, predict: %f'%(y_val[i], model.predict([X_val[i]])[0][0] ))
	
	return history, model

def evaluate(model, X, y, score_function):
	
	# Predict
	y_predict = []
	for i in range(len(X)):
		y_predict.append(model.predict([X[i]])[0][0])
	
	# Return score
	score = score_function(y, y_predict)
	return score
	
def predict(model, X):
	return model.predict([X])[0][0]

def predict_mul(model, X):
	return [model.predict([x])[0][0] for x in X]

