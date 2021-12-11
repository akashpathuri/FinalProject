import utility
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go

class perceptron:
    def __init__(self, image_size, feature_size, classfications):
        self.image_size = image_size
        self.feature_size = feature_size
        self.classfications = classfications
        self.number_of_weights = (self.image_size[0] * self.image_size[0]) // (self.feature_size * self.feature_size) + 1

        self.perceptron_weights = []
        for digit in range(classfications):
            digit_weight = []
            digit_weight.append(random.randint(0,5))
            for i in range(1, self.number_of_weights):
                weight = random.randint(0,100)
                digit_weight.append(weight)
            self.perceptron_weights.append(digit_weight)
            digit_weight = []
  
    def train_dataSet(self, images, image_labels):
        #Training loop
        training_results = []
        training_gradient = []
        
        epochs = 50
        for epoch in tqdm(range(epochs)):
            for i, image in enumerate(images):
                image_result = self.run_perceptron(image, image_labels[i], epoch > 0)
                if image_result == image_labels[i]:
                    training_results.append(1)
                else: 
                    training_results.append(0)
            training_gradient.append(np.mean(training_results) * 100)
            training_results = []
        plt.plot(training_gradient)
        plt.title("Accuracy of Face Perceptron over Training Cycle")
        plt.show()

    def test_dataSet(self, images, image_labels):
        testing_results = []
        for i, image in enumerate(images):
            image_result = self.run_perceptron(image, image_labels[i], False)
            if image_result == image_labels[i]:
                testing_results.append(1)
            else: 
                testing_results.append(0)
        
        fig = go.Figure(go.Indicator(
		mode = "gauge+number",
		value = np.mean(testing_results) * 100,
		title = {'text': "Accuracy"}))
        
        fig.show()
        
    
    def train_perceptron(self,features, image_label, image_result):
        if (image_result != image_label):
            self.perceptron_weights[image_result][0] -= 1
            self.perceptron_weights[image_label][0] += 1
            for i, feature in enumerate(features):
                self.perceptron_weights[image_result][i + 1] -= feature
                self.perceptron_weights[image_label][i + 1] += feature
                
    def activation(self, weighted_sums):
        max_sum = weighted_sums[0]
        max_digit = 0
        for digit, weighted_sum in enumerate(weighted_sums):
            if weighted_sum>max_sum:
                max_sum = weighted_sum
                max_digit = digit
        return max_digit
    
    def run_perceptron(self, image, image_label, training):
        features = utility.get_feature_values(image, self.feature_size)
        weighted_sums = []
        for digit_weights in self.perceptron_weights:
            weighted_sum = digit_weights[0]
            for i, feature in enumerate(features):
                weighted_sum += feature * digit_weights[i + 1]
            weighted_sums.append(weighted_sum)
        image_result = self.activation(weighted_sums)
        if (training):
            self.train_perceptron(features, image_label, image_result)
            
        return image_result
    
   
	