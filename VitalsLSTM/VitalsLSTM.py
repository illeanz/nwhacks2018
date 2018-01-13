import pandas as pd
import numpy as np
import argparse

from matplotlib import pyplot

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser()

#Training or Inference
parser.add_argument('--mode', type=str, default='infer',
                    help='train or infer')

parser.add_argument('--test', nargs='*', default=list(),
					help='List of previous vitals')

FLAGS = parser.parse_args()

# Look_back distance and # of features
LOOK_BACK = 5
N_FEATURES = 4

def pre_proc(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()

	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names

	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)

	return agg

def model_save(model):
	#Saving model (JSON) and weights (HD5) to disk
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	print("Saved model to disk")

def train():
	# Load Vitals.csv training dataset
	dataset = pd.read_csv('Vitals.csv', header=0, index_col=0)
	values = dataset.values

	# Cast all values to float
	values = values.astype('float32')
	# Pre-process data
	training = pre_proc(values, LOOK_BACK, 1)

	# Divide into training and test sets
	new_values = training.values
	n_obvals = LOOK_BACK * N_FEATURES
	train = new_values[:9601, :]
	test = new_values[9601:, :]
	# Divide into inputs and outputs
	train_X, train_y = train[:, :n_obvals], train[:, -N_FEATURES]
	test_X, test_y = test[:, :n_obvals], test[:, -N_FEATURES]
	# Reshape into [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], LOOK_BACK, N_FEATURES))
	test_X = test_X.reshape((test_X.shape[0], LOOK_BACK, N_FEATURES))

	#Encode train_y, test_y one-hot
	train_y = to_categorical(train_y)
	test_y = to_categorical(test_y)

	# Network Architecture
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(4, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	history = model.fit(train_X, train_y, epochs=50, batch_size=64, validation_data=(test_X, test_y), verbose=2,
						shuffle=False)
	# plot history
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	pyplot.show()

	model_save(model)
	print('Finished Training')

def model_load():
    # Reload model and parameters
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    trained_model = model_from_json(loaded_model_json)
    trained_model.load_weights("model.h5")
    print("Loaded model from disk")

    return trained_model

def predict(model, currTest):
	# make a prediction
	currTest = np.reshape(currTest, (1, LOOK_BACK, N_FEATURES))
	yhat = model.predict_classes(currTest)

	return yhat[0]

if (FLAGS.mode == 'train'):
	train()
elif (FLAGS.mode == 'infer'):
	currModel = model_load()
	currState = predict(currModel, FLAGS.test)
	print(currState)
else:
	raise ValueError('Unknown Command')


