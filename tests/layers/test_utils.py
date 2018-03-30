import numpy as np

from dlnpy.layers._utils import tensor_to_matrix

batch_size = 2
channels = 3
filters = 4

tensor = np.arange(batch_size*4*4*channels).reshape((batch_size, 4, 4, channels))

filter1 = np.array([
    [1, 1],
    [0, 0]
])
filter1 = np.tile(filter1, (channels, 1)).reshape(channels, 2, 2)
filter1 = filter1.transpose(1, 2, 0)

filter2 = np.array([
    [1, 0],
    [1, 0]
])
filter2 = np.tile(filter2, (channels, 1)).reshape(channels, 2, 2)
filter2 = filter2.transpose(1, 2, 0)

filter3 = np.array([
    [0, 1],
    [0, 1]
])
filter3 = np.tile(filter3, (channels, 1)).reshape(channels, 2, 2)
filter3 = filter3.transpose(1, 2, 0)

filter4 = np.array([
    [0, 0],
    [1, 1]
])
filter4 = np.tile(filter4, (channels, 1)).reshape(channels, 2, 2)
filter4 = filter4.transpose(1, 2, 0)

kernel = np.concatenate((filter1, filter2, filter3, filter4), axis=-1)
kernel = kernel.reshape(2, 2, filters, channels)
kernel = kernel.transpose((0, 1, 3, 2))
print(kernel[:, :, 0, 0])
print(kernel[:, :, 1, 0])
print(kernel[:, :, 2, 0])
print(kernel[:, :, 0, 1])
print(kernel[:, :, 1, 1])
print(kernel[:, :, 2, 1])
print(kernel[:, :, 0, 2])
print(kernel[:, :, 1, 2])
print(kernel[:, :, 2, 2])
print(kernel[:, :, 0, 3])
print(kernel[:, :, 1, 3])
print(kernel[:, :, 2, 3])

kernel_col = kernel.transpose((3, 2, 0, 1)).reshape(filters, -1)

col = tensor_to_matrix(tensor, (2, 2), (2, 2), 'valid', rank=2)

output = np.dot(kernel_col, col)
output = output.reshape((filters, 2, 2, batch_size))
print(output[0, :, :, 0])
print(output[1, :, :, 0])
