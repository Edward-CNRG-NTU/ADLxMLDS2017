import scipy.misc as misc
import numpy as np
import glob

filenames = glob.glob("./celebA/img_align_celeba/*.jpg")

N = len(filenames)
img_size = (64, 64, 3)
x_data = np.zeros([N, *img_size], dtype=np.uint8)

for i in range(N):
    img = misc.imread(filenames[i])
    x_data[i] = misc.imresize(img, size=img_size[:2])

np.save('celeba_data.npy', x_data)

# x_data = np.load('x_data.npy')

print(x_data.shape)

misc.imshow(x_data[-2])
