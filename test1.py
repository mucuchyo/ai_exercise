import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

dense = Dense(units = 1, input_shape = [1])
model = Sequential([dense])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1, 0, 1, 2, 3], dtype = float)
ys = np.array([-3, -1, 1, 3, 5], dtype = float)

model.fit(xs, ys,epochs = 500)
print(model.predict(np.array([10.0, 1 ,100000])))
print(format(dense.get_weights()))