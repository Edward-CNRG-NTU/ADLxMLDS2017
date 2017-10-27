# import tensorflow
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Flatten, Merge
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Masking
from keras.layers import Dense, Dropout, Reshape
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
import numpy as np
import sys

from data_processing import *
from sample_splitter import *
from sequence_composer import *
from data_augmentation import *


def make_model(time_step, data_dim, num_classes, adam_clip=0.):
    model_cnn_1 = Sequential()
    model_cnn_1.add(Reshape((time_step, data_dim, 1), input_shape=(time_step, data_dim)))
    model_cnn_1.add(Conv2D(32, (1, int(data_dim / 2)), padding='valid', activation='tanh'))
    model_cnn_1.add(MaxPooling2D(pool_size=(1, 2)))
    model_cnn_1.add(TimeDistributed(Flatten()))
    print(model_cnn_1.summary())

    model__cnn_2 = Sequential()
    model__cnn_2.add(Reshape((time_step, data_dim, 1), input_shape=(time_step, data_dim)))
    model__cnn_2.add(Conv2D(16, (1, 8), padding='valid', activation='tanh'))
    model__cnn_2.add(MaxPooling2D(pool_size=(1, 2)))
    model__cnn_2.add(Conv2D(16, (1, 8), padding='valid', activation='tanh'))
    model__cnn_2.add(MaxPooling2D(pool_size=(1, 2)))
    model__cnn_2.add(Conv2D(16, (1, 8), padding='valid', activation='tanh'))
    model__cnn_2.add(MaxPooling2D(pool_size=(1, 2)))
    model__cnn_2.add(TimeDistributed(Flatten()))
    print(model__cnn_2.summary())

    model = Sequential()
    model.add(Merge([model_cnn_1, model__cnn_2], mode='concat'))
    model.add(Bidirectional(LSTM(data_dim * 4, recurrent_dropout=0.1, dropout=0.2, return_sequences=True)))
    model.add(LSTM(num_classes, recurrent_dropout=0.1, dropout=0.2, return_sequences=True, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(clipvalue=adam_clip) if adam_clip > 0 else Adam(),
                  metrics=['accuracy'])
    print(model.summary())
    return model



# mode = 'train'
mode = 'predict'
try:
    data_dir = sys.argv[1]
    output_file = sys.argv[2]
except IndexError:
    print('usage: python3 cnn_model.py data_dir output_file')
    exit(0)

phone_full_dict = load_phone_dict(data_dir)
fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data,\
mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data = load_from_text(data_dir)
train_label = prepare_label_from_text(data_dir, phone_full_dict, fbank_train_id, mfcc_train_id)

# model constant
max_sequence = 200
batch_size = 64

if mode == 'train' or mode == 'continue':
    X_raw = np.concatenate((mfcc_train_data, fbank_train_data), axis=1)
    Y_raw = train_label
    sample_id = mfcc_train_id

    sample_indexer = make_sliding_indexer(X_raw.shape[0], max_sequence, int(max_sequence * 2 / 3))

    # reshape
    X_all = get_sample_by_indexer(X_raw, sample_indexer, padding='last')
    Y_all = get_label_by_indexer(Y_raw, sample_indexer, padding='last')
    print(X_all.shape, Y_all.shape)

    data_dim = X_all.shape[2]
    num_classes = Y_all.shape[2]

    if mode == 'continue':
        model = make_model(max_sequence, data_dim, num_classes, adam_clip=0.01)
        model.load_weights('cnn_model.hdf5')
    else:
        model = make_model(max_sequence, data_dim, num_classes, adam_clip=0.5)

    # save checkpoint
    callbacks_list = []
    callbacks_list += [ModelCheckpoint(filepath='cnn_model_{val_acc:.2f}.hdf5', monitor='val_acc',
                                       verbose=1, save_best_only=True, mode='max')]
    callbacks_list += [EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='min')]

    model.fit([X_all, X_all], Y_all, epochs=100, batch_size=batch_size, shuffle=False,
              validation_split=0.1, callbacks=callbacks_list)

elif mode == 'predict':
    X_raw = np.concatenate((mfcc_test_data, fbank_test_data), axis=1)
    sample_id = mfcc_test_id

    sample_indexer, sample_info_list = parse_sample_by_name(sample_id, 780)

    # reshape
    X_test = get_sample_by_indexer(X_raw, sample_indexer, padding='last')

    model = make_model(780, 108, 49)
    model.load_weights('cnn_model.hdf5')
    print(model.summary())

    Y_test = model.predict_classes([X_test, X_test], batch_size=batch_size)

    sample_dict = post_processing(Y_test, sample_info_list, phone_full_dict)

    export_result(sample_dict, file_name=output_file)
