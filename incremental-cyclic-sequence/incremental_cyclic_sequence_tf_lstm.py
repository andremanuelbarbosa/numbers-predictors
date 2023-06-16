import random

import numpy as np
import tensorflow as tf

cyclic_size: int = 4
features_size: int = 1
sequence_size: int = 5
data_size: int = sequence_size * cyclic_size
data: list[int] = []
for _ in range(0, data_size, sequence_size):
    for j in range(1, sequence_size + 1):
        data.append(j)

training_size: float = 0.8
training_data_x: list[list[int]] = []
training_data_y: list[int] = []
evaluation_data_x: list[list[int]] = []
evaluation_data_y: list[int] = []
for i in range(sequence_size, data_size):
    x = data[i - sequence_size:i]
    y = data[i]
    if random.random() < training_size:
        training_data_x.append(x)
        training_data_y.append(y)
    else:
        evaluation_data_x.append(x)
        evaluation_data_y.append(y)
training_data_x = np.array(training_data_x)
training_data_y = np.array(training_data_y)
training_data_x = training_data_x.reshape((training_data_x.shape[0], training_data_x.shape[1], features_size))
evaluation_data_x = np.array(evaluation_data_x)
evaluation_data_y = np.array(evaluation_data_y)

units = sequence_size * 10
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units, activation='relu', input_shape=(sequence_size, features_size)))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())

model.fit(training_data_x, training_data_y, epochs=max(1000, data_size))

print(model.evaluate(evaluation_data_x, evaluation_data_y))


def predict_number(sequence: list[int], expected: int):
    prediction_x = np.array(sequence)
    prediction_x = prediction_x.reshape((1, sequence_size, features_size))
    prediction = model.predict(prediction_x, verbose=0)
    prediction_number = round(prediction[0][0])
    print(f'predicted number after sequence {sequence} was {prediction_number}, expected was {expected}')


predict_number([1, 2, 3, 4, 5], 1)
predict_number([2, 3, 4, 5, 1], 2)
predict_number([3, 4, 5, 1, 2], 3)
predict_number([4, 5, 1, 2, 3], 4)
predict_number([5, 1, 2, 3, 4], 5)
