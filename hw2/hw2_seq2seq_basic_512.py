# import tensorflow
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Masking, Lambda
from keras.layers import Input, Embedding, Dense, Dropout, Reshape
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import numpy as np
import random
import time
import json
import sys
import os


def load_train_data(data_dir):
    train_label = json.load(open(data_dir + 'training_label.json', 'r', encoding='utf-8'))
    train_label_dict = {}
    all_texts = []
    for d in train_label:
        text = ['BOS ' + t + ' EOS' for t in d["caption"]]
        train_label_dict[d["id"]] = text
        all_texts += text
    # print(train_label_dict)

    tokenizer = Tokenizer(num_words=3000,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(all_texts)

    train_data_dict = {}
    train_id_list = []
    train_data_list = []
    train_label_list = []
    train_data_dir = data_dir + '/training_data/feat/'
    for filename in os.listdir(train_data_dir):
        # train_data_dict[filename[:-4]] = np.load(train_data_dir + filename)
        train_id_list += [filename[:-4]]
        train_data_list += [np.load(train_data_dir + filename)]
        train_label_list += [tokenizer.texts_to_sequences(train_label_dict[filename[:-4]])]
        # temp = tokenizer.texts_to_sequences(train_label_dict[filename[:-4]])
        # for t in temp:
        #     if max(t) == 2999:
        #         print(t)
        # print(train_id_list, train_data_list, train_label_list); exit()
    train_data_np = np.concatenate(train_data_list, axis=0)

    global _mean, _std
    _mean = np.mean(train_data_np, axis=0)
    _std = np.std(train_data_np, axis=0)
    _std[np.where(_std == 0.0)] = 1.
    train_data_np = ((train_data_np - _mean) / (2 * _std)).reshape([-1, 80, 4096])

    np.save('_mean', _mean)
    np.save('_std', _std)

    return train_id_list, train_data_np, train_label_list, tokenizer


def load_test_data(data_dir, tokenizer):
    test_label = json.load(open(data_dir + 'testing_label.json', 'r', encoding='utf-8'))
    test_label_dict = {}
    for d in test_label:
        text = ['BOS ' + t + ' EOS' for t in d["caption"]]
        test_label_dict[d["id"]] = text
    # print(test_label_dict)

    train_data_dict = {}
    test_id_list = []
    test_data_list = []
    test_label_list = []
    test_data_dir = data_dir + '/testing_data/feat/'
    for filename in os.listdir(test_data_dir):
        # train_data_dict[filename[:-4]] = np.load(test_data_dir + filename)
        test_id_list += [filename[:-4]]
        test_data_list += [np.load(test_data_dir + filename)]
        test_label_list += [tokenizer.texts_to_sequences(test_label_dict[filename[:-4]])]

    test_data_np = np.concatenate(test_data_list, axis=0)
    test_data_np = ((test_data_np - _mean) / (2 * _std)).reshape([-1, 80, 4096])

    return test_id_list, test_data_np, test_label_list


batch_size = 50


def data_generator(train_data_np, train_label_list, batch_size):
    last_position = 0
    while True:
        start = last_position if last_position < train_data_np.shape[0] else 0
        last_position = min(start + batch_size, train_data_np.shape[0])
        X = train_data_np[start:last_position]
        Y_list = [captions[random.randrange(len(captions))] for captions in train_label_list[start:last_position]]
        Y = pad_sequences(Y_list, maxlen=50 + 1, padding='post', value=0)
        Y_onehot = np.stack([np_utils.to_categorical(y, num_classes=3000) for y in Y], axis=0)
        Y_onehot[np.where(Y == 0)] = np.zeros([1, 3000])
        yield [X, Y_onehot[:, :-1]], Y_onehot[:, 1:]


def make_train_model(latent_dim=1024, adam_clip=0.):
    encoder_inputs = Input(shape=(80, 4096))
    encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, 3000))
    # embedding = Embedding(input_dim=3000, output_dim=3000, mask_zero=False,
    #                       weights=[np.identity(3000)], input_length=50, trainable=False)
    # embedding_output = embedding(decoder_inputs)
    masked_output = Masking(mask_value=0.)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, name='decoder_lstm')
    decoder_outputs = decoder_lstm(masked_output, initial_state=encoder_states)
    decoder_dense = Dense(3000, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    training_model.compile(optimizer=Adam(clipvalue=adam_clip) if adam_clip > 0 else Adam(),
                           loss='categorical_crossentropy', metrics=['accuracy'])
    training_model.summary()
    # plot_model(training_model, to_file='training_model.png')

    return training_model


def train_model(training_model, train_data_gen, test_data_gen, epochs=40):
    callbacks_list = [ModelCheckpoint(filepath='model', monitor='val_acc', verbose=1,
                                      save_weights_only=True, save_best_only=True, mode='max')]
    # callbacks_list += [TensorBoard(log_dir='logs', histogram_freq=5, batch_size=batch_size, write_graph=True,
    #                                write_grads=False, write_images=False, embeddings_freq=0,
    #                                embeddings_layer_names=None, embeddings_metadata=None)]
    training_model.fit_generator(train_data_gen, int(1450 / batch_size), epochs=epochs,
                                 validation_data=test_data_gen, validation_steps=1,
                                 class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=True,
                                 shuffle=True, initial_epoch=0, callbacks=callbacks_list)

    training_model.save_weights('trained_model')


def load_inference_models(latent_dim=1024):
    encoder_inputs = Input(shape=(80, 4096))
    encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.load_weights('model', by_name=True)

    decoder_inputs = Input(shape=(None, 3000))
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_dense = Dense(3000, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    decoder_model.load_weights('model', by_name=True)

    return encoder_model, decoder_model


def inference_sample(sample, tokenizer, encoder_model, decoder_model):
    BOS_token = tokenizer.word_index['bos']
    EOS_token = tokenizer.word_index['eos']
    target = np_utils.to_categorical(BOS_token, num_classes=3000).reshape([1, 1, -1])

    state = encoder_model.predict(sample)

    decoded_sentence = []
    while True:
        output_v, h, c = decoder_model.predict([target] + state)
        token = np.argmax(output_v[0, -1, :])
        state = [h, c]

        target = np_utils.to_categorical(token, num_classes=3000).reshape([1, 1, -1])
        # target = (output_v + target) / 2
        # target = output_v

        decoded_sentence += [token]

        if token == EOS_token or len(decoded_sentence) > 50:
            break

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    reverse_word_map[0] = ''
    reverse_word_map[EOS_token] = '.'

    return ' '.join([reverse_word_map[i] for i in decoded_sentence])


# mode = 'train'
# mode = 'continue'
mode = 'predict'
latent_dim = 512
# data_dir = '../MLDS_hw2_data/'
try:
    data_dir = sys.argv[1]
    output_filename = sys.argv[2]
except IndexError:
    print('Usage: python hw2_seq2seq_basic_512.py [the data directory] [output_file]')
    exit(0)

train_id_list, train_data_np, train_label_list, tokenizer = load_train_data(data_dir)
test_id_list, test_data_np, test_label_list = load_test_data(data_dir, tokenizer)
train_data_gen = data_generator(train_data_np, train_label_list, batch_size)
test_data_gen = data_generator(test_data_np, test_label_list, 100)

if mode == 'train':
    training_model = make_train_model(latent_dim)
    train_model(training_model, train_data_gen, test_data_gen, epochs=100)
elif mode == 'continue':
    training_model = make_train_model(latent_dim, adam_clip=0.01)
    training_model.load_weights('model', by_name=True)
    train_model(training_model, train_data_gen, test_data_gen, epochs=100)
elif mode == 'predict':
    encoder_model, decoder_model = load_inference_models(latent_dim)

    all_lines = open('special_mission_id.txt', 'r', encoding='utf-8').readlines()
    with open(output_filename, 'w', encoding='utf-8') as f:
        for filename in all_lines:
            filename = filename.strip(' \n')
            data = np.load(data_dir + 'testing_data/feat/' + filename + '.npy')
            data = ((data - _mean) / (2 * _std)).reshape([-1, 80, 4096])
            result = inference_sample(data, tokenizer, encoder_model, decoder_model)
            f.write(filename + ',' + result + '\n')

    # print(inference_sample(test_data_np[1].reshape([-1, 80, 4096]), tokenizer, encoder_model, decoder_model))
    # print(test_id_list[1])

# print(next(train_data_gen))
# print(next(train_data_gen))
