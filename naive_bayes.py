import utility
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go

class naive_bayes:
    def __init__(self, image_size, feature_size, classfications):
        self.image_size = image_size
        self.feature_size = feature_size
        self.classfications = classfications
        self.number_of_features = (image_size[0] * image_size[1]) // (feature_size * feature_size)
        
        self.bayes_network = []
        for c in range(classfications):
            classfier_probabilities = []
            for f in range(self.number_of_features):
                feature_probabilities = [0.0]*(feature_size*feature_size + 1)
                classfier_probabilities.append(feature_probabilities)
            self.bayes_network.append(classfier_probabilities)
        
        self.prior_probabilities = [0.0]*(classfications)
        #self.sample_size = sum(self.prior_probabilities)
        self.sample_size = 0
  
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
        
    
    def train_perceptron(self, features, image_label):
        self.sample_size += self.classfications
        self.prior_probabilities[image_label] += self.classfications
        for i, feature in enumerate(features):
            self.bayes_network[image_label][i][feature] += self.classfications
            
    def activation(self,weighted_probabilities):
        max_probablity = weighted_probabilities[0]
        max_result = 0
        for value, probablity in enumerate(weighted_probabilities):
            if probablity>max_probablity:
                max_probablity = probablity
                max_result = value
        return max_result
    
    def run_perceptron(self, face_image, image_label, training):
        features = utility.get_feature_values(face_image, self.feature_size)
        weighted_probabilities = []
        total_sample = float(self.sample_size)
        if(total_sample==0):
            total_sample = 1
        for i, prior in enumerate(self.prior_probabilities):
            if(prior == 0):
                prior = 0.0001
            total_probablity = float(prior / total_sample);
            for j, feature in enumerate(features):
                #print(i, j, feature)
                feature_probablity = self.bayes_network[i][j][feature]
                if(feature_probablity == 0):
                    feature_probablity = 0.0001
                total_probablity *= float(feature_probablity / total_sample)
            weighted_probabilities.append(total_probablity)
            
        image_result = self.activation(weighted_probabilities)
        if (training):
            self.train_perceptron(features, image_label)
            
        return image_result
    
   
	