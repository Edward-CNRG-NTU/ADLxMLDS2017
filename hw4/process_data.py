import scipy.misc as misc
import numpy as np

N = 33431
img_size = (64, 64, 3)
x_data = np.zeros([N, *img_size], dtype=np.uint8)

for i in range(N):
    img = misc.imread('faces/%d.jpg' % i)
    if img.shape == (96, 96, 3):
        x_data[i] = misc.imresize(img, size=img_size[:2])
    else:
        print('wrong shape at %d' % i)

np.save('x_data.npy', x_data)

# x_data = np.load('x_data.npy')

print(x_data.shape)

misc.imshow(x_data[-1])
