import random
import pickle
import cv2
import os
import numpy as np
from imutils import paths


ImagePaths = list(paths.list_images("VegetablesFruits\Fresh"))
print(len(ImagePaths))
ImagePaths = ImagePaths[:1630] + list(paths.list_images("VegetablesFruits\Stale"))
print(len(ImagePaths))
random.shuffle(ImagePaths)

data = []
labels = []
for imagepath in ImagePaths:
    image = cv2.imread(imagepath)
    image = cv2.resize(image, (32,32))
    data.append(image)
    label = imagepath.split(os.path.sep)[-2]
    if label == "Stale":
        label = [1,0]
    else:
        label = [0,1]
    labels.append(label)
print(labels)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

with open("data.pickle", 'wb') as f:
    pickle.dump(data, f)
print("Data seved")

with open("labels.pickle", 'wb') as f:
    pickle.dump(labels, f)
print("Labels seved")