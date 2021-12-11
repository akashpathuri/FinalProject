import utility

face_images = utility.load_data_file("facedata/facedatatrain")
face_image_labels = utility.load_data_labels("facedata/facedatatrainlabels")

# for line in face_images[0]:
# 	print(line)

# features = utility.extract_features(face_images[0], 10)
# for feature in features:
# 	for line in feature:
# 		print(line)
# 	print()

print(utility.get_feature_values(face_images[0]))