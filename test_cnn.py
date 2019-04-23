import cv2
import keras
import argparse
import numpy as np
from keras.models import load_model

#Default size
heightResize = 240
widthResize  = 320

if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description='CNN Test for Star Wars')

	### Positional arguments
	parser.add_argument('-i', '--inputDir', required=True, help="Path to the input dir to the image for detection")
	parser.add_argument('-m', '--modelDir', default="models/", help="Path to folder dir of trained model")
	parser.add_argument('-n', '--nameModel', default="modelTrain.h5", help="Name of the trained model for detection")

	args = vars(parser.parse_args())

	srcDir	  = (args["inputDir"])
	modelDir  = (args["modelDir"])
	nameModel = (args["nameModel"])

	#Load Model
	modelRecognition = keras.models.load_model(modelDir + nameModel)

	#Read image
	src = cv2.imread(srcDir)
	src = cv2.resize(src, (heightResize, widthResize)) 

	cv2.imshow("Input image", src)
	cv2.waitKey()

	#Normalization
	src = (src)/255.0
	srcTest = []
	srcTest.append(src)
	srcTest = np.array(srcTest)

	#Prediction
	prediction = modelRecognition.predict(srcTest)

	#Display Results (We only have two classes)
	if (prediction < 0.5):
		print("\n\n\n*************************")
		print("Luke Skywalker")
		print("*************************")
	else:
		print("\n\n\n*************************")
		print("Darth Vader ")
		print("*************************")