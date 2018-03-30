import numpy as np

from dlnpy.models import Sequential
from dlnpy.layers import Dense, Sigmoid, Softmax
from dlnpy.optimizers import SGD
from dlnpy.losses import categorical_crossentropy
from dlnpy.utils import to_categorical, train_test_split
from dlnpy.datasets import digits


(digits_x0, digits_y0), (digits_x1, digits_y1) = digits.load_data()
digits_x = np.concatenate((digits_x0, digits_x1))
digits_y = np.concatenate((digits_y0, digits_y1))

digits_x = digits_x / 16
digits_y = to_categorical(digits_y, 10)

train_x, valid_x, train_y, valid_y = train_test_split(
    digits_x, digits_y, test_size=.2
)

model = Sequential()
model.add(Dense(50, input_shape=(64, )))
model.add(Sigmoid())
model.add(Dense(10))
model.add(Softmax())

model.summary()

optimizer = SGD(lr=0.01, momentum=.9)
loss = categorical_crossentropy

model.compile(optimizer, loss, metrics='accuracy')
model.fit(
    train_x, train_y,
    batch_size=128, epochs=100, validation_data=(valid_x, valid_y),
)
