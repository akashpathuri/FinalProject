from naive_bayes import naive_bayes
from perceptron import perceptron
from KNN import KNN
import utility
import numpy as np
import random
from tqdm import tqdm
from timeit import default_timer as timer

def face_classfication(model, percentage):
	image_size = [70, 60]
	feature_size = 5
	classfications = 2
 
	images = utility.load_data_file("facedata/facedatatrain", image_size[0])
	image_labels = utility.load_data_labels("facedata/facedatatrainlabels")

	images_zip = list(zip(images, image_labels))

	images = utility.load_data_file("facedata/facedatatest", image_size[0])
	image_labels = utility.load_data_labels("facedata/facedatatestlabels")

	number_of_data_points = int(percentage * len(images_zip))

	results = []
	timing = []
	for i in range(5):
		images_sample_zip = random.sample(images_zip, number_of_data_points)
		images_sample_zip = [list(t) for t in zip(*images_sample_zip)]

		if(model == "naive_bayes"):
			face_model = naive_bayes(image_size, feature_size, classfications)
		elif model == "perceptron":
			face_model = perceptron(image_size, feature_size, classfications)
		elif model == "knn":
			face_model = KNN(image_size, feature_size, classfications, 10)

		start = timer()
		face_model.train_dataSet(images_sample_zip[0], images_sample_zip[1])
		end = timer()
		timing.append(end - start) 

		results.append(face_model.test_dataSet(images, image_labels))

	# print("Points:\t", number_of_data_points)
	# print("Timing:\t", np.mean(timing))
	# print("Mean:\t", np.mean(results))
	# print("Stdev:\t", np.std(results))
	print(percent * 100, number_of_data_points, np.mean(timing), np.mean(results), np.std(results),sep=", ")
 
def digit_classfication(model, percentage):
	image_size = [28, 28]
	feature_size = 4
	classfications = 10

	images = utility.load_data_file("digitdata/trainingimages", image_size[0])
	image_labels = utility.load_data_labels("digitdata/traininglabels")

	images_zip = list(zip(images, image_labels))	

	images = utility.load_data_file("digitdata/testimages", image_size[0])
	image_labels = utility.load_data_labels("digitdata/testlabels")

	number_of_data_points = int(percentage * len(images_zip))

	results = []
	timing = []
	for i in range(5):
		images_sample_zip = random.sample(images_zip, number_of_data_points)
		images_sample_zip = [list(t) for t in zip(*images_sample_zip)]

		if(model == "naive_bayes"):
			digit_model = naive_bayes(image_size, feature_size, classfications)
		elif model == "perceptron" :
			digit_model = perceptron(image_size, feature_size, classfications)
		elif model == "knn":
			digit_model = KNN(image_size, feature_size, classfications, 10)

		start = timer()
		digit_model.train_dataSet(images_sample_zip[0], images_sample_zip[1])
		end = timer()
		timing.append(end - start)

		results.append(digit_model.test_dataSet(images, image_labels))

	# print("Points:\t", number_of_data_points)
	# print("Timing:\t", np.mean(timing))
	# print("Mean:\t", np.mean(results))
	# print("Stdev:\t", np.std(results))
	print(percent * 100, number_of_data_points, np.mean(timing), np.mean(results), np.std(results),sep=", ")
		
print("Percentage, Points, Timing, Accuracy, Std. Dev")		
for p in range(1,11):
	percent = p / 10
	# print("---------------------------------------")
	# print("Perceptron Face (percentage:", percent, ")")
	face_classfication("perceptron", percent)
	# print()

	# print("Naive Bayes Face (percentage:", percent, ")")
	# face_classfication("perceptron", percent)
	# print()

	# print("KNN Face (percentage:", percent, ")")
	# face_classfication("knn", percent)
	# print()

	# print("Perceptron Digit (percentage:", percent, ")")
	# digit_classfication("perceptron", percent)
	# print()

	# print("Naive Bayes Digit (percentage:", percent, ")")
	# digit_classfication("naive_bayes", percent)
	# print()

	# print("KNN Digit (percentage:", percent, ")")
	# digit_classfication("knn", percent)
	