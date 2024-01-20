import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


#ЧТЕНИЕ ДАТАСЕТА И ЛЕЙБЛОВ

with open("data.pickle",'rb') as f:
    data = pickle.load(f)
print("Data loaded")


with open("labels.pickle",'rb') as f:
    labels = pickle.load(f)
print("Labels loaded")


(trainX, testX, trainY, testY) = train_test_split(data, labels, 
                                                  test_size=0.15,
                                                  random_state=42)

#ДВУХМЕРНАЯ МОДЕЛЬ

model = Sequential()
model.add(Conv2D(32,(3,3), padding="same", input_shape=(32,32,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding="same", activation="relu"))
model.add(Conv2D(64,(3,3), padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

#КОМПИЛЯЦИЯ МОДЕЛИ

INIT_LR = 0.01
#opt = SGD(learning_rate=INIT_LR)
opt = Adam(learning_rate=0.001,beta_1=0.9, beta_2=0.999, amsgrad=False)


model.compile(loss="binary_crossentropy", optimizer=opt, 
              metrics=["accuracy"])
print("Model compiled")
#model.summary()


#ОБУЧЕНИЕ

EPOCHS = 20

# early_stopping = EarlyStopping(monitor="val_accuracy", patience=2, verbose=1)
# checkpointer = ModelCheckpoint(filepath="EasyNet_{epoch:2d}.hdf5", verbose=1, save_best_only=True)


H = model.fit(trainX,trainY, validation_data=(testX,testY),
              epochs=EPOCHS, batch_size=32,
              shuffle=True)
              #callbacks=[early_stopping,checkpointer])
print("Model trained")

#ГРАФИК ФУНКЦИЙ И СОХРАНЕНИЕ МОДЕЛИ И ГРАФИКА

predictions = model.predict(testX, batch_size=32)
print(predictions) 

print(classification_report(testY.argmax(axis=1),
                             predictions.argmax(axis=1), target_names = ("Stale","Fresh")))

N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="vall_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Results")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy/")
plt.legend()
plt.savefig("Loss.png")
model.save("EasyNet.model")