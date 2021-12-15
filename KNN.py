import utility
import random
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from collections import Counter

class KNN:
    def __init__(self, image_size, feature_size, classfications, k=3):
        self.image_size = image_size
        self.feature_size = feature_size
        self.classfications = classfications
        self.number_of_weights = (self.image_size[0] * self.image_size[0]) // (self.feature_size * self.feature_size) + 1
        self.k = k
  
        self.neighbors = []
        self.neighbor_labels = []
  
    def train_dataSet(self, images, image_labels):
        #Training loop
        training_results = []
        training_gradient = []
        
        epochs = 1
        # for epoch in tqdm(range(epochs)):
        for epoch in range(epochs):
            for i, image in enumerate(images):
                image_result = self.run_perceptron(image, image_labels[i], True)
                if image_result == image_labels[i]:
                    training_results.append(1)
                else: 
                    training_results.append(0)
            training_gradient.append(np.mean(training_results) * 100)
            training_results = []
        # plt.plot(training_gradient)
        # plt.title("Accuracy of Face Perceptron over Training Cycle")
        # plt.show()

    def test_dataSet(self, images, image_labels):
        testing_results = []
        # for i, image in enumerate(tqdm(images)):
        for i, image in enumerate(images):
            image_result = self.run_perceptron(image, image_labels[i], False)
            if image_result == image_labels[i]:
                testing_results.append(1)
            else: 
                testing_results.append(0)
        
        return np.mean(testing_results) * 100

        # fig = go.Figure(go.Indicator(
		# mode = "gauge+number",
		# value = np.mean(testing_results) * 100,
		# title = {'text': "Accuracy"}))
        
        # fig.show()
                
    def activation(self, nearest_labels):
        relevant_k = len(nearest_labels) if self.k > len(nearest_labels) else self.k
        nearest_label = Counter(nearest_labels[:self.k]).most_common(1)
        return nearest_label[0][0]
    
    def neighbor_distances(self, features):
        distances = []
        for neighbor in self.neighbors:
            differences = 0
            for i, feature in enumerate(features):
                differences += (neighbor[i] - feature)**2
            distance = np.sqrt(differences)
            distances.append(distance)
        return distances
        
    
    def run_perceptron(self, image, image_label, training):
        features = utility.get_feature_values(image, self.feature_size)
        if(training):
            self.neighbors.append(features)
            self.neighbor_labels.append(image_label)
            return image_label
        distances = self.neighbor_distances(features)
        distances = np.argsort(distances)
        #print(len(distances), distances) 
        nearest_labels = [self.neighbor_labels[index] for index in distances ]
        image_result = self.activation(nearest_labels)
            
        return image_result
    
   
	