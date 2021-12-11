from naive_bayes import naive_bayes
from perceptron import perceptron
from KNN import KNN
import utility

def face_classfication(model):
	image_size = [70, 60]
	feature_size = 5
	classfications = 2
	if(model == "naive_bayes"):
		face_model = naive_bayes(image_size, feature_size, classfications)
	elif model == "perceptron":
		face_model = perceptron(image_size, feature_size, classfications)
	elif model == "knn":
		face_model = KNN(image_size, feature_size, classfications, 10)
 
	images = utility.load_data_file("facedata/facedatatrain", image_size[0])
	image_labels = utility.load_data_labels("facedata/facedatatrainlabels")
	face_model.train_dataSet(images, image_labels)

	images = utility.load_data_file("facedata/facedatatest", image_size[0])
	image_labels = utility.load_data_labels("facedata/facedatatestlabels")
	face_model.test_dataSet(images, image_labels)
 
def digit_classfication(model):
	image_size = [28, 28]
	feature_size = 4
	classfications = 10
	if(model == "naive_bayes"):
		digit_model = naive_bayes(image_size, feature_size, classfications)
	elif model == "perceptron" :
		digit_model = perceptron(image_size, feature_size, classfications)
	elif model == "knn":
		digit_model = KNN(image_size, feature_size, classfications, 10)
  
	images = utility.load_data_file("digitdata/trainingimages", image_size[0])
	image_labels = utility.load_data_labels("digitdata/traininglabels")
	digit_model.train_dataSet(images, image_labels)

	images = utility.load_data_file("digitdata/testimages", image_size[0])
	image_labels = utility.load_data_labels("digitdata/testlabels")
	digit_model.test_dataSet(images, image_labels)
 
#digit_classfication("naive_bayes")
#face_classfication("naive_bayes")
#digit_classfication("perceptron")
#face_classfication("perceptron")
digit_classfication("knn")
#face_classfication("knn")