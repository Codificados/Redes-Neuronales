import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import numpy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

neuronas = [64]
densas = [2]
convpoo = [1]
drop = [0]

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x / 255.0
y = numpy.array(y)

for neurona in neuronas:
    for conv in convpoo:
        for densa in densas:
            for d in drop:
                NAME = "RedConv_n{}_cl{}_d{}_dropout{}".format(neurona,conv,densa,d)
                tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

                model = Sequential()
                model.add(Conv2D(neurona, (3,3), input_shape = x.shape[1:]))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size = (2,2)))

                if d == 1:
                    model.add(Dropout(0.2))

                for i in range(conv):
                    model.add(Conv2D(neurona, (3,3)))
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size = (2,2)))

                model.add(Flatten())

                for i in range(densa):
                    model.add(Dense(neurona))
                    model.add(Activation("relu"))

                model.add(Dense(1))
                model.add(Activation('sigmoid'))

                model.compile(loss="binary_crossentropy",
                              optimizer="adam",
                              metrics=['accuracy'])

                model.fit(x,y, batch_size = 32, epochs = 10, validation_split = 0.3, callbacks=[tensorboard])
                model.save("models/{}".format(NAME))