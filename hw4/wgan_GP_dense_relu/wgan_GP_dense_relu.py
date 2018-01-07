import time
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras import layers
from keras import optimizers
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU


x_data = np.load('x_data.npy')
x_data = x_data / 255.
y_embedded = np.load('y_onehot.npy')

hair_embedded = np.load('hair_onehot.npy')
eyes_embedded = np.load('eyes_onehot.npy')
eyes_hair_embedded = np.load('eyes_hair_onehot.npy')


N = 33431
noise_dim = 40
embed_dim = 24
laten_dim = 256
class_dim = 1
img_size = (64, 64, 3)

d_lr = 0.0002
gan_lr = 0.0002

penalty_factor = 10

epoch_max = 1000
batch_size = 64

project_dir = './wgan_GP_dense_selu/'

d_train_factor = 1

# _mean = np.mean(x_data)
# _std = np.std(x_data)
# x_data = (x_data - _mean) / _std
# np.save(project_dir + '_mean.npy', _mean)
# np.save(project_dir + '_std.npy', _std)
# print(_mean, _std)


def build_generator_model():
    K.set_learning_phase(0)

    txt_input = layers.Input(shape=(embed_dim,))
    noise_input = layers.Input(shape=(noise_dim,))

    net = layers.Concatenate(axis=-1)([noise_input, txt_input])

    net = layers.Dense(laten_dim * 16 * 16, kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Reshape((16, 16, laten_dim))(net)

    net = layers.Conv2DT3ranspose(laten_dim // 4, [3, 3], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2DTranspose(laten_dim // 8, [5, 5], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2D(laten_dim // 16, [5, 5], strides=[1, 1], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)

    net = layers.Conv2D(img_size[2], [1, 1], strides=[1, 1], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation('sigmoid')(net)

    model = Model(inputs=[noise_input, txt_input], outputs=net)
    # model.load_weights(project_dir + 'generator_model')
    model.summary()

    return model


def build_discriminator_model():
    txt_input = layers.Input(shape=(embed_dim,))
    img_input = layers.Input(shape=img_size)

    net = layers.Conv2D(laten_dim // 4, [5, 5], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(img_input)
    net = layers.Activation(activation='relu')(net)
    net = layers.AlphaDropout(0.2)(net)

    net = layers.Conv2D(laten_dim // 2, [5, 5], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)
    net = layers.AlphaDropout(0.2)(net)

    net = layers.Conv2D(laten_dim, [3, 3], strides=[2, 2], padding='same', kernel_initializer='lecun_normal')(net)
    net = layers.Activation(activation='relu')(net)
    net = layers.AlphaDropout(0.2)(net)

    net = layers.Conv2D(laten_dim, [3, 3], strides=[2, 2], padding='same')(net)
    net = layers.Activation(activation='relu')(net)
    net = layers.AlphaDropout(0.2)(net)

    net = layers.Lambda(
        lambda x: K.concatenate([x[0], K.reshape(K.repeat(x[1], n=16), shape=(-1, 4, 4, embed_dim))], axis=-1)
    )([net, txt_input])

    net = layers.Conv2D(laten_dim * 2, [4, 4], strides=[1, 1], padding='valid')(net)
    net = layers.Activation(activation='relu')(net)
    net = layers.AlphaDropout(0.2)(net)

    net = layers.Flatten()(net)

    net = layers.Dense(class_dim, activation='linear', kernel_initializer='lecun_normal')(net)

    model = Model(inputs=[img_input, txt_input], outputs=net)
    # model.load_weights(project_dir + 'discriminator_model')
    model.summary()

    return model


def build_train_fn(generator_model, discriminator_model):
    true_img_input = layers.Input(shape=img_size)
    true_txt_input = layers.Input(shape=(embed_dim,))
    wrong_txt_input = layers.Input(shape=(embed_dim,))
    noise_input = layers.Input(shape=(noise_dim,))

    fake_img_ph = generator_model([noise_input, true_txt_input])

    loss_true_case = K.mean(discriminator_model([true_img_input, true_txt_input]))
    loss_fake_case = K.mean(discriminator_model([fake_img_ph, true_txt_input]))
    loss_wrong_txt = K.mean(discriminator_model([true_img_input, wrong_txt_input]))

    mix_factor_ph = K.random_uniform(shape=())

    mixed_img_ph = mix_factor_ph * true_img_input + (1. - mix_factor_ph) * fake_img_ph

    mixed_grad = K.gradients(discriminator_model([mixed_img_ph, true_txt_input]), [mixed_img_ph, true_txt_input])

    norm_mixed_grad = K.sqrt(K.sum(K.square(mixed_grad[0]), axis=[1, 2, 3]) + K.sum(K.square(mixed_grad[1]), axis=[1]))
    grad_penalty = K.mean(K.square(norm_mixed_grad - 1.))

    obj_d = loss_true_case - 0.5 * loss_fake_case - 0.5 * loss_wrong_txt - penalty_factor * grad_penalty
    update_d = optimizers.Adam(lr=d_lr, beta_1=0., beta_2=0.9).get_updates(params=discriminator_model.trainable_weights,
                                                                           loss=-obj_d)
    train_fn_d = K.function(inputs=[true_img_input, true_txt_input, noise_input, wrong_txt_input],
                            outputs=[obj_d],
                            updates=update_d)

    obj_g = loss_fake_case
    update_g = optimizers.Adam(lr=gan_lr, beta_1=0., beta_2=0.9).get_updates(params=generator_model.trainable_weights,
                                                                             loss=-obj_g)
    train_fn_g = K.function(inputs=[noise_input, true_txt_input],
                            outputs=[obj_g],
                            updates=update_g)

    return train_fn_g, train_fn_d


def save_test_img(epoch):
    gen_img = g_model.predict([fixed_z_, fixed_txt_])

    gen_img = gen_img * 255.

    fig, ax = plt.subplots(11, 12, figsize=(11, 12))
    for i in range(11):
        for j in range(12):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
            ax[i, j].cla()
            ax[i, j].imshow(gen_img[i * 12 + j].astype(np.uint8))

    fig.tight_layout()
    fig.text(0.5, 0.003, 'Epoch %d' % epoch, ha='center')
    plt.savefig(project_dir + 'epoch%d.jpg' % epoch, dpi=128)
    plt.close()

fixed_z_ = np.random.normal(size=(11 * 12, noise_dim))
fixed_txt_ = eyes_hair_embedded


def save_model_checkpoint(epoch, gan_acc, d_acc):
    log_str = '%s epoch: %d gan_acc: %.4f d_acc: %.4f ' \
              % (time.strftime("%Y%m%d-%H%M%S"), epoch, gan_acc, d_acc)

    d_model.save_weights(project_dir + 'discriminator_model_%s' % log_str)
    g_model.save_weights(project_dir + 'generator_model_%s' % log_str)


g_model = build_generator_model()
d_model = build_discriminator_model()
train_g, train_d = build_train_fn(g_model, d_model)


for epoch in range(epoch_max):
    print('=================================== Epoch %d ===================================' % epoch)
    gan_loss = gan_acc = d_loss = d_acc = 0.

    for batch in range(N // batch_size):
        random_index_all = np.random.permutation(N)
        random_index_true = random_index_all[:batch_size]
        random_index_other = random_index_all[batch_size:2 * batch_size]

        true_img = x_data[random_index_true]
        true_txt = y_embedded[random_index_true]

        wrong_txt = y_embedded[random_index_other]

        noise = np.random.normal(size=(batch_size, noise_dim))

        d_loss = train_d([true_img, true_txt, noise, wrong_txt])[0]

        gan_loss = train_g([noise, true_txt])[0]

        log_str = '%s epoch: %d batch: %d gan_loss: %.4f gan_acc: %.4f d_loss: %.4f d_acc: %.4f '\
                  % (time.strftime("%Y%m%d-%H%M%S"), epoch, batch, gan_loss, 0., d_loss, 0.)

        print(log_str, file=open(project_dir + 'log', 'a'))
        print(log_str)

    d_model.save_weights(project_dir + 'discriminator_model')
    g_model.save_weights(project_dir + 'generator_model')

    if epoch % 3 == 0:
        save_test_img(epoch)

    if epoch % 10 == 0:
        save_model_checkpoint(epoch, gan_acc, d_acc)

