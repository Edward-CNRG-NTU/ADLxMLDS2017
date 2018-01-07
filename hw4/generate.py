import numpy as np
import scipy.misc as misc
import sys

from keras import layers
from keras.models import Model
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU

try:
    test_text = sys.argv[1]
except IndexError:
    test_text = 'testing_text.txt'

hair_embedded = np.load('hair_onehot.npy')
eyes_embedded = np.load('eyes_onehot.npy')

hair_tags = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
             'green hair', 'red hair', 'purple hair', 'pink hair',
             'blue hair', 'black hair', 'brown hair', 'blonde hair']

eyes_tags = ['gray eyes', 'black eyes', 'orange eyes',
             'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
             'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

num_classes = len(hair_tags) + len(eyes_tags) + 1

np.random.seed(72)

noise_dim = 40
embed_dim = 24
laten_dim = 256
class_dim = 1
img_size = (64, 64, 3)

N = 5

def build_generator_model():
    K.set_learning_phase(1)

    txt_input = layers.Input(shape=(embed_dim,))
    noise_input = layers.Input(shape=(noise_dim,))

    net = layers.Concatenate(axis=-1)([noise_input, txt_input])

    net = layers.Dense(laten_dim * 16 * 16, kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Reshape((16, 16, laten_dim))(net)

    net = layers.Conv2DTranspose(laten_dim // 4, [3, 3], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2DTranspose(laten_dim // 8, [5, 5], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2D(laten_dim // 16, [5, 5], strides=[1, 1], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2D(img_size[2], [1, 1], strides=[1, 1], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation('sigmoid')(net)

    model = Model(inputs=[noise_input, txt_input], outputs=net)
    model.load_weights('wgan_GP_dense_relu/generator_model_20171230-232858 epoch: 970 gan_acc: 0.0000 d_acc: 0.0000 ')
    # model.summary()

    return model


g_model = build_generator_model()

with open(test_text, 'r') as f:
    for line in f.readlines():
        index, text = line.split(',')

        text_onehot = np.zeros([1, num_classes])
        for i, hair_tag in enumerate(hair_tags):
            if hair_tag in text:
                text_onehot += hair_embedded[i]
        for i, eyes_tag in enumerate(eyes_tags):
            if eyes_tag in text:
                text_onehot += eyes_embedded[i]

        text_onehot = np.repeat(text_onehot, N, axis=0)
        noise = np.random.normal(size=(N, noise_dim))

        imgs = g_model.predict([noise, text_onehot])

        for i, img in enumerate(imgs):
            misc.imsave('samples/sample_%s_%d.jpg' % (index, i + 1), img)
