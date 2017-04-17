import sys

# import tensorflow as tf
import keras
from keras import backend as K
# from keras.utils import generic_utils
# # from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Bidirectional, Reshape, Activation, TimeDistributed
from keras.optimizers import SGD
# from keras.utils.np_utils import to_categorical
# from keras.layers.normalization import BatchNormalization
import numpy as np
from numpy import genfromtxt

def preprocessing():
	f = open('train.txt', 'w')
	count = 0
	with open('train.csv', 'r') as reader:
		for line in reader:
			if count > 0:
				words = line.strip().split(',')
				string = ' '.join(words[1:])
				f.write(string+'\n')
			count += 1
	f.close()

	train_data = np.loadtxt('train.txt')
	print train_data.shape

	f = open('extra.txt', 'w')
	count = 0
	with open('extra_unsupervised_data.csv', 'r') as reader:
		for line in reader:
			words = line.strip().split(',')
			string = ' '.join(words)
			f.write(string+'\n')

	f.close()

	extra_data = np.loadtxt('extra.txt')
	print extra_data.shape


	f = open('test.txt', 'w')
	count = 0
	with open('test.csv', 'r') as reader:
		for line in reader:
			if count > 0:
				words = line.strip().split(',')
				string = ' '.join(words)
				f.write(string+'\n')
			count += 1
	f.close()

	test_data = np.loadtxt('test.txt')
	print test_data.shape

def first_model():
	model = Sequential()
	model.add(Dense(1000, activation='relu', input_dim=1875))
	#model.add(Dropout(0.5))
	#model.add(Dense(1000, activation='relu'))
	model.add(Dropout(0.4))
	
	model.add(Dense(9))
	model.add(Activation('softmax'))
	# model.add(Dense(1))
	sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	return model

def produce_labels(iteration):
	f = open('labels_extra_hot.txt', 'w')
	with open('pred_labels'+str(iteration)+'.txt', 'r') as reader:
		for line in reader:
			lis = [0] * 9
			label = int(line.strip()[0])
			lis[label] = 1
			f.write(' '.join(map(str,lis)))
			f.write('\n')
	f.close()

def concatenate_data_files():
	filenames = ['train.txt', 'extra.txt']
	with open('train_all.txt', 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)

def concatenate_label_files():
	filenames = ['labels_hot.txt', 'labels_extra_hot.txt']
	with open('labels_all.txt', 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)	

if __name__ == '__main__':
	#preprocessing()
	train_data = np.loadtxt('train.txt')
	test_data = np.loadtxt('test.txt')
	extra_data = np.loadtxt('extra.txt')
	train_labels = np.loadtxt('labels_hot.txt')

	#concatenate_data_files()
	train_all_data = np.loadtxt('train_all.txt')

	fmodel = first_model()
	fmodel.fit(train_data, train_labels, nb_epoch=100, batch_size=128)
	pred_labels = fmodel.predict_classes(extra_data, batch_size=1)
	np.savetxt('pred_labels0.txt', pred_labels)

	

	for i in xrange(100):
		fmodel = first_model()
		produce_labels(i)
		concatenate_label_files()
		train_all_labels = np.loadtxt('labels_all.txt')
		fmodel.fit(train_all_data, train_all_labels, nb_epoch=100, batch_size=128)
		pred_labels = fmodel.predict_classes(extra_data, batch_size=1)
		pred_test_labels = fmodel.predict_classes(test_data, batch_size=1)
		np.savetxt('pred_labels_test'+str(i)+'.txt', pred_test_labels)
		np.savetxt('pred_labels'+str(i+1)+'.txt', pred_labels)


	pred_test_labels = fmodel.predict_classes(test_data, batch_size=1)
	np.savetxt('pred_labels_test.txt', pred_test_labels)


