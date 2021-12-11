def load_data_file(filename, height):
	file = open(filename)
	images = []
	line_number = 0
	current_image = []

	for line in file.readlines():
		line = line[:-1]
		characters = [char for char in line]
		current_image.append(characters)
		line_number += 1
	
		if (line_number == height):
			images.append(current_image)
			line_number = 0
			current_image = []
	file.close()
	return images

def load_data_labels(filename):
	file = open(filename)
	labels = []
	for line in file.readlines():
		line = line[:-1]
		labels.append(int(line))
	file.close()
	return labels

def get_feature_values(face_image, feature_size):
	feature_values = []
	features = extract_features(face_image, feature_size)
	for feature in features:
		feature_value = compute_features(feature)
		feature_values.append(feature_value)
	return feature_values

def extract_features(face_image, feature_size):
	features = []
	for current_y in range(len(face_image) // feature_size):
		for current_x in range(len(face_image[0]) // feature_size):
			feature = []
			for y in range(feature_size):
				line = []
				for x in range(feature_size):
					line.append(face_image[current_y * feature_size + y][current_x * feature_size + x])
				feature.append(line)
			features.append(feature)
	return features

def compute_features(feature):
	count = 0
	for line in feature:
		for char in line:
			if (char != ' '):
				count += 1
	return count