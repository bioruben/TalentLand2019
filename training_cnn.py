import os
import cv2
import glob
import argparse
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

#Setup variables
batchSize      = 32
epochs         = 1
validationSize = 0.2 # 20 percent (0.2)
heightResize   = 240
widthResize    = 320

#Input Data
foldersNames = ['Luke', 'Vader']
imgFormats   = ['jpeg', 'png', 'jpg']

#Read Data for clasification
def buildDataset( foldersNames, imgFormats):
    dataset = []
    labels  = []
    label   = 0 

    for folder in foldersNames:
	    for i in range(len(imgFormats)):
	        dirString =  args["input_dir"] + "/" + folder + "/*." + imgFormats[i]
	        for filename in glob.glob(dirString):
	            src = cv2.imread(filename)
	            src = cv2.resize(src, (heightResize, widthResize)) 
	            dataset.append(src)    
	            labels.append(label)
	    label+= 1
    return dataset, labels



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Image Classifier Training')

	### Positional arguments
	parser.add_argument('-i', '--input_dir', default="data/", help="Input dir for the data folder")
	parser.add_argument('-m', '--modelDir', default="models/", help="Path to folder dir of trained model")
	parser.add_argument('-n', '--nameModel', default="modelTrain.h5", help="Name of the trained model for detection")

	args = vars(parser.parse_args())

	modelDir  = (args["modelDir"])
	nameModel = (args["nameModel"])
	saveDir   = modelDir + nameModel

	#Read Data for Training
	[dataset, labels] = buildDataset( foldersNames, imgFormats )
	X = np.array(dataset)
	Y = np.array(labels)
	
	#Normalize input data
	X = X/255.0
	
	#CNN
	print("*********Training*********")

	#Feature extraction	
	model = Sequential()	
	model.add(Conv2D(64 , (3,3) , input_shape = X.shape[1:] ))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size= (2,2)))
	model.add(Conv2D(64 , (3,3) ))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size= (2,2)))

	#Classification
	model.add(Flatten())
	model.add(Dense(64))

	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	#Initialize RMS prop optimizer
	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) #'adam', 

	#Model configuration
	model.compile(loss = 'binary_crossentropy', 
		optimizer = opt,
		metrics = ['accuracy'])

	model.fit(X, Y, batch_size = batchSize, epochs = epochs, validation_split = validationSize)
	model.summary()

	#Save model and weights
	if not os.path.isdir(modelDir):
	    os.makedirs(modelDir)
	model.save(saveDir)
	print('Saved trained model at %s ' % saveDir)