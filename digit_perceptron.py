import utility
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go


def train_perceptron(perceptron_weights, features, digit_image_label, image_result):
	if (image_result != digit_image_label):
		perceptron_weights[image_result][0] -= 1
		perceptron_weights[digit_image_label][0] += 1
		for i, feature in enumerate(features):
			perceptron_weights[image_result][i + 1] -= feature
			perceptron_weights[digit_image_label][i + 1] += feature

	
	return perceptron_weights

def activation(weighted_sums):
	max_sum = weighted_sums[0]
	max_digit = 0
	for digit, weighted_sum in enumerate(weighted_sums):
		if weighted_sum>max_sum:
			max_sum = weighted_sum
			max_digit = digit
	return max_digit
		

def run_perceptron(digit_image, digit_image_label, perceptron_weights, feature_size, training):
	features = utility.get_feature_values(digit_image, feature_size)
	weight_sums = []
	for digit_weights in perceptron_weights:
		weight_sum = digit_weights[0]
		for i, feature in enumerate(features):
			weight_sum += feature * digit_weights[i + 1]
		weight_sums.append(weight_sum)
		
	image_result = activation(weight_sums)
	if (training):
		perceptron_weights = train_perceptron(perceptron_weights, features, digit_image_label, image_result)
	
	return image_result, perceptron_weights

def main():
	image_height = 28
	image_width = 28 
	feature_size = 4
	number_of_weights = (image_height * image_width) // (feature_size * feature_size) + 1
	number_of_digits = 10
 
	perceptron_weights = []
	for digit in range(number_of_digits):
		digit_weight = []
		digit_weight.append(random.randint(0,5))
		for i in range(1, number_of_weights):
			weight = random.randint(0,100)
			digit_weight.append(weight)
		perceptron_weights.append(digit_weight)
		digit_weight = []

	#Training loop
	digit_images = utility.load_data_file("digitdata/trainingimages", image_height)
	digit_image_labels = utility.load_data_labels("digitdata/traininglabels")
	training_results = []
	training_gradient = []
	
	epochs = 50
	for epoch in tqdm(range(epochs)):
		for i, digit_image in enumerate(digit_images):
			image_result, perceptron_weights = run_perceptron(digit_image, digit_image_labels[i], perceptron_weights, feature_size, epoch > 0)

			if image_result == digit_image_labels[i]:
				training_results.append(1)
			else: 
				training_results.append(0)
		training_gradient.append(np.mean(training_results) * 100)
		training_results = []
	plt.plot(training_gradient)
 
	#Testing loop
	digit_images = utility.load_data_file("digitdata/testimages", image_height)
	digit_image_labels = utility.load_data_labels("digitdata/testlabels")
	testing_results = []
	for i, digit_image in enumerate(digit_images):
		image_result, _ = run_perceptron(digit_image, digit_image_labels[i], perceptron_weights, feature_size, False)
		if image_result == digit_image_labels[i]:
			testing_results.append(1)
		else: 
			testing_results.append(0)

	fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = np.mean(testing_results) * 100,
    title = {'text': "Accuracy"}))

	fig.show()
	plt.title("Accuracy of Digit Perceptron over Training Cycle")
	plt.show()


main()