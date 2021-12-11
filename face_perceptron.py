import utility
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go

def activation(weighted_sum):
	return 1 if weighted_sum > 0 else 0

def train_perceptron(perceptron_weights, features, face_image_label, image_result):
	if (image_result and not face_image_label):
		train_factor = -1
	elif (not image_result and face_image_label):
		train_factor = 1
	else:
		train_factor = 0
	
	if train_factor != 0:
		perceptron_weights[0] += train_factor
		for i, feature in enumerate(features):
			perceptron_weights[i + 1] += feature * train_factor
	
	return perceptron_weights

def run_perceptron(face_image, face_image_label, perceptron_weights, feature_size, training):
	features = utility.get_feature_values(face_image, feature_size)
	weight_sum = perceptron_weights[0]
	for i, feature in enumerate(features):
		weight_sum += feature * perceptron_weights[i + 1]
	
	image_result = activation(weight_sum)
	if (training):
		perceptron_weights = train_perceptron(perceptron_weights, features, face_image_label, image_result)
	
	return image_result, perceptron_weights

def main():
	image_height = 70
	image_width = 60 
	feature_size = 5
	number_of_weights = (image_height * image_width) // (feature_size * feature_size) + 1

	perceptron_weights = []
	perceptron_weights.append(random.randint(-1,1))
	for i in range(1, number_of_weights):
		weight = random.randint(-100,100)
		perceptron_weights.append(weight)

	#Training loop
	face_images = utility.load_data_file("facedata/facedatatrain", image_height)
	face_image_labels = utility.load_data_labels("facedata/facedatatrainlabels")
	training_results = []
	training_gradient = []
	print(perceptron_weights)
	
	epochs = 50
	for epoch in tqdm(range(epochs)):
		for i, face_image in enumerate(face_images):
			image_result, perceptron_weights = run_perceptron(face_image, face_image_labels[i], perceptron_weights, feature_size, epoch > 0)

			if image_result == face_image_labels[i]:
				training_results.append(1)
			else: 
				training_results.append(0)
		training_gradient.append(np.mean(training_results) * 100)
		training_results = []

	print(perceptron_weights)
	plt.plot(training_gradient)
 
	#Testing loop
	face_images = utility.load_data_file("facedata/facedatatest", image_height)
	face_image_labels = utility.load_data_labels("facedata/facedatatestlabels")
	testing_results = []
	print(perceptron_weights)
	for i, face_image in enumerate(face_images):
		image_result, _ = run_perceptron(face_image, face_image_labels[i], perceptron_weights, feature_size, False)
		if image_result == face_image_labels[i]:
			testing_results.append(1)
		else: 
			testing_results.append(0)

	fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = np.mean(testing_results) * 100,
    title = {'text': "Accuracy"}))

	fig.show()
	plt.title("Accuracy of Face Perceptron over Training Cycle")		
	plt.show()


main()