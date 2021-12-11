from numpy import double
import utility
from decimal import *

face_images = utility.load_data_file("facedata/facedatatrain", 70)
face_image_labels = utility.load_data_labels("facedata/facedatatrainlabels")

# for line in face_images[0]:
# 	print(line)

# features = utility.extract_features(face_images[0], 10)
# for feature in features:
# 	for line in feature:
# 		print(line)
# 	print()

#print(utility.get_feature_values(face_images[0], 5))
getcontext().prec = 10
divide = (Decimal(1) / Decimal(10000))
print(divide)