import numpy as np

from dlnpy.layers.utils import image2column, column2image

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
# Reshape into (output_h, output_w, channels, filters)
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

# Reshape into (filters, channels, output_h, output_w)
ckernel = kernel.transpose((3, 2, 0, 1)).reshape(filters, -1)

ctensor = image2column(tensor, (2, 2), (2, 2), 'valid', rank=2)

cconv = np.dot(ctensor, ckernel.T)
# Reshape into (batch_size, output_h, output_w, filters)
conv = cconv.reshape((batch_size, 2, 2, filters))
print('')
print(conv[0, :, :, 0])
print(conv[0, :, :, 1])
print(conv[0, :, :, 2])
print(conv[0, :, :, 3])
print(conv[1, :, :, 0])
print(conv[1, :, :, 1])
print(conv[1, :, :, 2])
print(conv[1, :, :, 3])

# Reshape (batch_size, output_h, output_w, filters)
gy = np.ones_like(conv)
gy = gy.reshape(-1, filters)
gkernel = np.dot(ctensor.T, gy)
gkernel = gkernel.transpose((1, 0)).reshape(filters, channels, 2, 2)
gkernel = gkernel.transpose((2, 3, 1, 0))
# print(gkernel[:, :, 0, 0])
# print(gkernel[:, :, 1, 0])
# print(gkernel[:, :, 2, 0])
# print(gkernel[:, :, 0, 1])
# print(gkernel[:, :, 1, 1])
# print(gkernel[:, :, 2, 1])
# print(gkernel[:, :, 0, 2])
# print(gkernel[:, :, 1, 2])
# print(gkernel[:, :, 2, 2])
# print(gkernel[:, :, 0, 3])
# print(gkernel[:, :, 1, 3])
# print(gkernel[:, :, 2, 3])
gbias = np.sum(gy, axis=0)

cgy = np.dot(gy, ckernel)

gtensor = column2image(cgy, tensor.shape, (2, 2), (2, 2), 'valid', rank=2)
print('')
print(gtensor[0, :, :, 0])
print(gtensor[0, :, :, 1])
print(gtensor[0, :, :, 2])
print(gtensor[1, :, :, 0])
print(gtensor[1, :, :, 1])
print(gtensor[1, :, :, 2])


cpool = ctensor.reshape(-1, 2 * 2)
cpool = np.max(cpool, axis=-1)
pool = cpool.reshape((batch_size, 2, 2, channels))
print('')
print(pool[0, :, :, 0])
print(pool[0, :, :, 1])
print(pool[0, :, :, 2])
print(pool[1, :, :, 0])
print(pool[1, :, :, 1])
print(pool[1, :, :, 2])
