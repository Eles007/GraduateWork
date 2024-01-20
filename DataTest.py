import cv2
import random
from imutils import paths
from keras.models import load_model

#ВЫВОД И ПРОВЕРКА

model = load_model("EasyNet.model")

ImagePaths = list(paths.list_images("TestPicture"))
random.shuffle(ImagePaths)

i=0
for imagepath in ImagePaths:
    image = cv2.imread(imagepath)
    resized = cv2.resize(image,(300,300))
    data = cv2.resize(image,(32,32))
    data = data.reshape((1,32,32,3))
    pred = model.predict(data)
    print(pred, i)
    i += 1
    if pred[0][0]>pred[0][1]:
        cv2.imshow("Stale",resized)
    else:
        cv2.imshow("Fresh",resized)
    cv2.waitKey(0)