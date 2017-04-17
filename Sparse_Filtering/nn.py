import numpy as np
import pickle
import sys

import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import generic_utils
# from keras.preprocessing import sequence
# from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers import Bidirectional, Reshape, Activation, TimeDistributed
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

if __name__ == '__main__':
	x_train = np.loadtxt('train_nn.txt')
	y_train = np.loadtxt('labels_hot.txt')
	x_test = np.loadtxt('test_nn.txt')
	#x_test = to_categorical(x_test)
	print(x_train.shape)
	print(y_train.shape)
	print(x_test.shape)
	model = Sequential()
	#model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(120,10,1)))
	#model.add(Conv2D(32, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dense(8000, activation='relu', input_dim=1200))
	model.add(Dropout(0.4))
	#model.add(Dense(1000, activation='relu'))
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.4))
	#model.add(Dense(1500, activation='sigmoid'))
	#model.add(Dropout(0.4))

	#model.add(Dense(1000, activation='relu'))
	#model.add(Dropout(0.4))
	model.add(Dense(9))
	model.add(Activation('softmax'))
	# model.add(Dense(1))
	#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	model.fit(x_train, y_train, nb_epoch=200, batch_size=128)
	pred_labels = model.predict_classes(x_test, batch_size=1)
	np.savetxt('pred_labels.txt', pred_labels)

