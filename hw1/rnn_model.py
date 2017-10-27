# import tensorflow
from keras import initializers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import LSTM, Bidirectional, TimeDistributed, Masking
from keras.layers import Dense, Dropout, Reshape
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
import numpy as np
import sys

from data_processing import *
from sample_splitter import *
from sequence_composer import *
from data_augmentation import *


# normalize
def normalize(data, load, axis=0):
    if load is True:
        _mean = np.load(model_dir + '_mean.npy')
        _std = np.load(model_dir + '_std.npy')
    else:
        _mean = np.mean(data, axis=axis)
        _std = np.std(data, axis=axis)
        np.save(model_dir + '_mean', _mean)
        np.save(model_dir + '_std', _std)
    return (data - _mean) / _std


def make_sliding_indexer(source_length, window_size, stride=1):
    return np.arange(window_size)[None, :] + np.arange(0, source_length - (window_size - 1), stride)[:, None]


def make_model(time_step, data_dim, num_classes, adam_clip=0.):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_step, data_dim)))
    model.add(Bidirectional(LSTM(data_dim * 2, recurrent_dropout=0.1, dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(data_dim * 2, recurrent_dropout=0.1, dropout=0.2, return_sequences=True)))
    model.add((LSTM(num_classes, recurrent_dropout=0.1, dropout=0.2, return_sequences=True, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue=adam_clip) if adam_clip > 0 else Adam(),
                  metrics=['accuracy'])
    print(model.summary())
    return model


def get_sample_by_indexer(X_raw, sample_indexer, padding='zero'):
    if padding == 'zero':
        X_padding = np.concatenate((X_raw, np.zeros((1, X_raw.shape[1]))), axis=0)
    elif padding == 'last':
        X_padding = X_raw
    return X_padding[sample_indexer]


def get_label_by_indexer(Y_raw, sample_indexer, padding='zero'):
    if padding == 'zero':
        Y_padding = np.concatenate((Y_raw, [len(set(Y_raw))]), axis=0)
    elif padding == 'last':
        Y_padding = Y_raw
    Y_padding = np_utils.to_categorical(Y_padding)
    return Y_padding[sample_indexer]


def post_processing(Y_test, sample_info_list, phone_full_dict):
    if Y_test.shape[0] != len(sample_info_list):
        print('Y_test, sample_info_list: length mismatch!', Y_test.shape, len(sample_info_list))
        return None

    sample_dict = {}
    for i in range(Y_test.shape[0]):
        sample_info = sample_info_list[i]

        for j in range(Y_test.shape[1]):
            try:
                character = phone_full_dict[Y_test[i, j]]['mapped_to']['char']
            except KeyError:
                character = '.'

            try:
                sample_dict[sample_info[0]]['raw'] += character
                if sample_dict[sample_info[0]]['reduced'][-1] != character:
                    sample_dict[sample_info[0]]['reduced'] += character
            except KeyError:
                sample_dict[sample_info[0]] = {}
                sample_dict[sample_info[0]]['raw'] = character
                sample_dict[sample_info[0]]['reduced'] = character

    for key in sample_dict.keys():
        sample_dict[key]['reduced'] = sample_dict[key]['reduced'].strip('L.')

    return sample_dict


def export_result(sample_dict):
    with open('sample.csv', 'r') as fr:
        sample_format = fr.readlines()

    with open(model_dir + 'result_reduced.csv', 'w') as fw:
        fw.write(sample_format[0])
        for line in sample_format[1:]:
            id = line.split(',')[0]
            try:
                fw.write(id + ',' + sample_dict[id]['reduced'] + '\n')
            except KeyError:
                print('id: %s not found!' % id)
                return

    with open(model_dir + 'result_raw.csv', 'w') as fw:
        fw.write(sample_format[0])
        for line in sample_format[1:]:
            id = line.split(',')[0]
            try:
                fw.write(id + ',' + sample_dict[id]['raw'] + '\n')
            except KeyError:
                print('id: %s not found!' % id)
                return


# mode = 'train'
# mode = 'predict'
try:
    mode = sys.argv[1]
except IndexError:
    mode = 'preview'
model_dir = __file__[:-3] + '/'

phone_full_dict = load_phone_dict()
# fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_text(phone_full_dict)
fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_binary()

# model constant
max_sequence = 200
batch_size = 160

if mode == 'train' or mode == 'continue':
    X_raw = np.concatenate((mfcc_train_data, fbank_train_data), axis=1)
    Y_raw = train_label
    print(X_raw.shape, Y_raw.shape)

    # augmentation
    for i in range(4):
        alpha = np.random.uniform(0.9, 1.1)
        aug_fbank, freq = logscale_spec(fbank_train_data, alpha=alpha)
        X_aug = np.concatenate((mfcc_train_data, aug_fbank), axis=1)
        X_raw = np.concatenate((X_raw, X_aug), axis=0)
        Y_raw = np.concatenate((Y_raw, train_label), axis=0)

    print(X_raw.shape, Y_raw.shape)

    X_raw = normalize(X_raw, load=False)
    # sample_indexer, sample_info_list = parse_sample_by_name(sample_id, max_sequence)
    sample_indexer = make_sliding_indexer(X_raw.shape[0], max_sequence, int(max_sequence * 0.8))

    # reshape
    X_all = get_sample_by_indexer(X_raw, sample_indexer, padding='last')
    Y_all = get_label_by_indexer(Y_raw, sample_indexer, padding='last')
    print(X_all.shape, Y_all.shape)

    data_dim = X_all.shape[2]
    num_classes = Y_all.shape[2]

    if mode == 'continue':
        model = make_model(max_sequence, data_dim, num_classes, adam_clip=0.01)
        model.load_weights(model_dir + 'model.hdf5')
    else:
        model = make_model(max_sequence, data_dim, num_classes, adam_clip=0.5)

    # save checkpoint
    callbacks_list = []
    callbacks_list += [ModelCheckpoint(filepath=model_dir + 'model_{val_acc:.2f}.hdf5', monitor='val_acc',
                                       verbose=1, save_best_only=True, mode='max')]
    callbacks_list += [
        TensorBoard(log_dir=model_dir + 'logs', histogram_freq=5, batch_size=batch_size, write_graph=True,
                    write_grads=False, write_images=False, embeddings_freq=0,
                    embeddings_layer_names=None, embeddings_metadata=None)]
    # tensorboard --logdir=/home/cnrg-ntu/PycharmProjects/MLDS/main_mfcc_fbank_s2s_crnn/logs

    callbacks_list += [EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='min')]

    model.fit(X_all, Y_all, epochs=100, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=callbacks_list)

elif mode == 'predict':
    X_raw = np.concatenate((mfcc_test_data, fbank_test_data), axis=1)
    Y_raw = None
    sample_id = mfcc_test_id

    X_raw = normalize(X_raw, load=True)

    sample_indexer, sample_info_list = parse_sample_by_name(sample_id, 780)

    # reshape
    X_test = get_sample_by_indexer(X_raw, sample_indexer, padding='last')

    model = make_model(780, 108, 48, adam_clip=0.001)
    model.load_weights(model_dir + 'model.hdf5')

    Y_test = model.predict_classes(X_test, batch_size=batch_size)

    np.save(model_dir + 'Y_test', Y_test)

    sample_dict = post_processing(Y_test, sample_info_list, phone_full_dict)

    export_result(sample_dict)

elif mode == 'preview':
    make_model(max_sequence, 108, 49)