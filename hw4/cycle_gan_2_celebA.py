import time
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal



x_data = np.load('x_data.npy')
x_data = x_data / 255.
p_data = np.load('celeba_data.npy')
p_data = p_data / 255.

N_1 = x_data.shape[0]
N_2 = p_data.shape[0]

laten_dim = 32
img_size = (64, 64, 3)

d_lr = 0.0002
g_lr = 0.0002

λ = 10

epoch_max = 300
batch_size = 8

project_dir = './cycle_gan_2_celebA/'

d_train_factor = 1

# def __conv_init(a):
#     print("conv_init", a)
#     k = RandomNormal(0, 0.02)(a) # for convolution kernel
#     k.conv_weight = True
#     return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)


def conv2d(f, *a, **k):
    return layers.Conv2D(f, kernel_initializer=conv_init, *a, **k)


def conv2d_T(f, *a, **k):
    return layers.Conv2DTranspose(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return layers.BatchNormalization(momentum=0.9, epsilon=1.01e-5, gamma_initializer=gamma_init)


def BASIC_D(ndf, n_hidden_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    _ = input_a = layers.Input(shape=img_size)
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='first')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, n_hidden_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", use_bias=False, name='hidden_%d' % layer)(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** n_hidden_layers, 8)
    # _ = layers.ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='hidden_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # _ = layers.ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=1, name='final', activation="sigmoid" if use_sigmoid else None)(_)

    return Model(inputs=[input_a], outputs=_)


def BASIC_G(ngf):
    _ = inputs = layers.Input(shape=img_size)

    _ = conv2d(ngf * 1, kernel_size=4, strides=1, padding="same", name='first')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d(ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False, name='hidden_%d' % 1)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d(ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False, name='hidden_%d' % 2)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d(ngf * 8, kernel_size=4, strides=1, padding="same", use_bias=False, name='hidden_%d' % 3)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d_T(ngf * 4, kernel_size=4, strides=2, padding="same", use_bias=False, name='hidden_%d' % 4)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d_T(ngf * 2, kernel_size=4, strides=2, padding="same", use_bias=False, name='hidden_%d' % 5)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d(ngf * 1, kernel_size=4, strides=1, padding="same", use_bias=False, name='hidden_%d' % 6)(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = conv2d(3, kernel_size=4, strides=1, padding="same", name='output')(_)
    _ = layers.Activation('sigmoid')(_)

    return Model(inputs=inputs, outputs=[_])


def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate


def D_loss(netD, real, fake, rec):
    loss_fn = lambda output, target: K.mean(K.binary_crossentropy(target, output))
    # loss_fn = lambda output, target: K.mean(K.abs(K.square(output - target)))
    output_real = netD([real])
    output_fake = netD([fake])
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))
    loss_D = loss_D_real+loss_D_fake
    loss_cyc = K.sqrt(K.mean(K.square(rec - real)))
    # loss_cyc = K.mean(K.abs(rec-real))
    return loss_D, loss_G, loss_cyc

D_A = BASIC_D(laten_dim, n_hidden_layers=4)
D_B = BASIC_D(laten_dim, n_hidden_layers=4)
D_A.summary()

G_A = BASIC_G(laten_dim)
G_B = BASIC_G(laten_dim)
G_A.summary()

from keras.utils import plot_model
plot_model(D_A, to_file=project_dir + 'D_model.png')
plot_model(G_A, to_file=project_dir + 'G_model.png')

real_A, fake_B, rec_A, cycleA_generate = cycle_variables(G_B, G_A)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(G_A, G_B)

loss_DA, loss_GA, loss_cycA = D_loss(D_A, real_A, fake_A, rec_A)
loss_DB, loss_GB, loss_cycB = D_loss(D_B, real_B, fake_B, rec_B)
loss_cyc = loss_cycA + loss_cycB
loss_G = loss_GA + loss_GB + λ * loss_cyc
loss_D = loss_DA + loss_DB

weightsD = D_A.trainable_weights + D_B.trainable_weights
weightsG = G_A.trainable_weights + G_B.trainable_weights

d_training_updates = optimizers.Adam(lr=d_lr, beta_1=0.5).get_updates(weightsD, [], loss_D)
train_D = K.function([real_A, real_B], [loss_DA / 2, loss_DB / 2], d_training_updates)
g_training_updates = optimizers.Adam(lr=g_lr, beta_1=0.5).get_updates(weightsG, [], loss_G)
train_G = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], g_training_updates)


def minibatch(data, batchsize):
    length = data.shape[0]
    idx = np.arange(length)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            random.shuffle(idx)
            i = 0
            epoch += 1
        rtn = [data[idx[j]] for j in range(i, i + size)]
        i += size
        tmpsize = yield epoch, np.float32(rtn)


def minibatchAB(dataA, dataB, batchsize):
    batchA = minibatch(dataA, batchsize)
    batchB = minibatch(dataB, batchsize)
    tmpsize = None
    while True:
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B


def save_test_img(epoch, imgs, row=6, col=8):
    imgs *= 255.
    imgs = imgs.clip(0., 255.).astype(np.uint8)

    fig, ax = plt.subplots(row, col, figsize=(row, col))
    for i in range(row):
        for j in range(col):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].cla()
            ax[i, j].imshow(imgs[i * col + j])

    fig.tight_layout()
    fig.text(0.5, 0.003, 'Epoch %d' % epoch, ha='center')
    plt.savefig(project_dir + 'epoch%d.jpg' % epoch, dpi=128)
    plt.close()


def showG(epoch, A, B):
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i: i + 1]]) for i in range(X.shape[0])])
        return r.swapaxes(0, 1)[:, :, 0]
    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A, B, rA[0], rB[0], rA[1], rB[1]])
    save_test_img(epoch, arr)


train_batch = minibatchAB(x_data, p_data, batch_size)

# _, A, B = next(train_batch)
# print(_, A.shape, B.shape)
# del A, B

t0 = time.time()
gen_iterations = 0
epoch = 0
errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

display_iters = 50

while epoch < epoch_max:
    last_epoch = epoch
    epoch, A, B = next(train_batch)
    errDA, errDB = train_D([A, B])
    errDA_sum += errDA
    errDB_sum += errDB

    # epoch, trainA, trainB = next(train_batch)
    errGA, errGB, errCyc = train_G([A, B])
    errGA_sum += errGA
    errGB_sum += errGB
    errCyc_sum += errCyc
    gen_iterations += 1
    if gen_iterations % display_iters == 0:
        log_str = '%s [%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc: %f sec: %.2f' % \
                  (time.strftime("%Y%m%d-%H%M%S"), epoch, epoch_max, gen_iterations,
                   errDA_sum/display_iters, errDB_sum/display_iters,
                   errGA_sum/display_iters, errGB_sum/display_iters,
                   errCyc_sum/display_iters, time.time()-t0)
        print(log_str)
        print(log_str, file=open(project_dir + 'log', 'a'))

        _, A, B = next(train_batch)
        showG(epoch, A, B)
        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

    if last_epoch != epoch:
        D_A.save_weights(project_dir + 'D_A_epoch%d' % epoch)
        D_B.save_weights(project_dir + 'D_B_epoch%d' % epoch)
        G_A.save_weights(project_dir + 'G_A_epoch%d' % epoch)
        G_B.save_weights(project_dir + 'G_B_epoch%d' % epoch)