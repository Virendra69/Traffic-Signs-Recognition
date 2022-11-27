from calendar import EPOCH
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
from PIL import Image

df_train = pd.read_csv('E:\VS Code Programs\Python_Codes\Deep_Learning\Projects\Traffic Signs Recognition\Images\Train.csv')
df_test = pd.read_csv('E:\VS Code Programs\Python_Codes\Deep_Learning\Projects\Traffic Signs Recognition\Images\Test.csv')

X = []
y = []

for c in range(43):
    path = os.path.join('E:\VS Code Programs\Python_Codes\Deep_Learning\Projects\Traffic Signs Recognition\Images\Train\\' + str(c))
    images = os.listdir(path)

    for i in images:
        try:
            img = Image.open(path + '\\' + i)
            img = img.resize((30, 30))
            img = np.array(img)/255
            X.append(img)
            y.append(c)
        except:
            print("Error Loading Image")

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

y_train = keras.utils.to_categorical(y_train, 43)
y_test = keras.utils.to_categorical(y_test, 43)

model = keras.Sequential([
    keras.layers.Conv2D(32, (5,5), activation = 'relu', input_shape = (30, 30, 3)),
    keras.layers.Conv2D(32, (5,5), activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(rate = 0.25),
    keras.layers.Conv2D(64, (5,5), activation = 'relu'),
    keras.layers.Conv2D(64, (5,5), activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(rate = 0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dropout(rate = 0.5),
    keras.layers.Dense(43, activation = 'softmax')
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train, y_train, epochs = 5, validation_data = (X_test, y_test))

model.save('traffic_classifier.h5')