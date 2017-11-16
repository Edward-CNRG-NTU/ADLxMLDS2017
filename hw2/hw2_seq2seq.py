# import tensorflow
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, GRU, Bidirectional, TimeDistributed, Masking, Lambda, Flatten, Permute
from keras.layers import Input, Embedding, Dense, Dropout, Reshape, Concatenate, RepeatVector, Add, Activation
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
import numpy as np
import random
import json
import sys
import os

# mode = 'train'
# mode = 'continue'
mode = 'predict'


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
        yield [X, Y_onehot[:, :-1]], Y_onehot[:, 1:]


def make_train_model(latent_dim=1024, adam_clip=0.):
    encoder_inputs = Input(shape=(80, 4096))
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    # encoder_states = [state_h, state_c]

    repeat_vector = RepeatVector(80)
    add_layer = Add()
    tanh_activate = Activation('tanh')
    flatten_layer = Flatten()
    softmax_activate = Activation('softmax')
    attention_dense_h = Dense(latent_dim, use_bias=True, name='attention_dense_h')
    attention_dense_e = Dense(latent_dim, use_bias=True, name='attention_dense_e')
    attention_dense = Dense(1, use_bias=False, name='attention_dense')

    def apply_attention(x):
        c = K.batch_dot(x[0], K.permute_dimensions(x[1], pattern=(0, 2, 1)), axes=(1, 2))
        return K.reshape(c, shape=(-1, 1, latent_dim))

    apply_attention_vector = Lambda(apply_attention, output_shape=(1, latent_dim), name='apply_attention')

    decoder_input = Input(shape=(50, 3000))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(3000, activation='softmax', name='decoder_dense')
    concat_layer = Concatenate(axis=-1)

    global teaching_rate_var
    teaching_rate_var = K.ones((), name='teaching_rate_v', dtype='float32')

    def teaching_rate(x):
        p = K.random_uniform((), minval=0.0, maxval=1.0) + teaching_rate_var
        # b = K.print_tensor(K.less_equal(p, 1.0), message='p:')
        # K.get_variable_shape(tr)
        return K.switch(K.less_equal(p, 1.0), x[0], x[1])

    teaching_rate_layer = Lambda(teaching_rate, output_shape=(1, 3000), name='teaching_rate')
    recurrent_input = Lambda(lambda x: x[:, 0:1, :], output_shape=(1, 3000), name='slice_init')(decoder_input)

    state = [state_h, state_c]
    output_list = []

    for t in range(50):
        input_at_t = Lambda(lambda a: a[:, t:t + 1, :], output_shape=(1, 3000), name='slice_%d' % t)(decoder_input)
        recurrent_input = teaching_rate_layer([input_at_t, recurrent_input])

        transformed_state_h = attention_dense_h(state_h)
        repeated_transformed_state_h = repeat_vector(transformed_state_h)
        transformed_encoder_outputs = attention_dense_e(encoder_outputs)

        attention_raw = add_layer([repeated_transformed_state_h, transformed_encoder_outputs])
        attention_pre = tanh_activate(attention_raw)
        attention_vector = softmax_activate(flatten_layer(attention_dense(attention_pre)))
        attention_context = apply_attention_vector([attention_vector, encoder_outputs])
        fused_input = concat_layer([recurrent_input, attention_context])
        output, state_h, state_c = decoder_lstm(fused_input, initial_state=state)
        dense_output = decoder_dense(output)

        recurrent_input = dense_output
        state = [state_h, state_c]
        output_list += [dense_output]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(output_list)

    training_model = Model([encoder_inputs, decoder_input], decoder_outputs)
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

    def update_tr(epoch, logs):
        global teaching_rate_var
        teaching_rate_var = K.update(teaching_rate_var, (1.0 * epoch) / epochs)
        print(K.eval(teaching_rate_var))

    callbacks_list += [LambdaCallback(on_epoch_begin=update_tr)]

    training_model.fit_generator(train_data_gen, int(1450 / batch_size), epochs=epochs,
                                 validation_data=test_data_gen, validation_steps=1,
                                 class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=True,
                                 shuffle=True, initial_epoch=0, callbacks=callbacks_list)

    training_model.save_weights('trained_model')
    # np.save('encoder_lstm', training_model.get_layer(name='encoder_lstm').get_weights())
    # np.save('attention_dense', training_model.get_layer(name='attention_dense').get_weights())
    # np.save('decoder_lstm', training_model.get_layer(name='decoder_lstm').get_weights())
    # np.save('decoder_dense', training_model.get_layer(name='decoder_dense').get_weights())


def load_inference_models(latent_dim=1024):
    encoder_inputs = Input(shape=(80, 4096))
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])
    encoder_model.load_weights('model_final', by_name=True)
    # encoder_model.get_layer(name='encoder_lstm').set_weights(np.load('encoder_lstm.npy'))

    repeat_vector = RepeatVector(80)
    add_layer = Add()
    tanh_activate = Activation('tanh')
    flatten_layer = Flatten()
    softmax_activate = Activation('softmax')
    attention_dense_h = Dense(latent_dim, use_bias=True, name='attention_dense_h')
    attention_dense_e = Dense(latent_dim, use_bias=True, name='attention_dense_e')
    attention_dense = Dense(1, use_bias=False, name='attention_dense')

    def apply_attention(x):
        c = K.batch_dot(x[0], K.permute_dimensions(x[1], pattern=(0, 2, 1)), axes=(1, 2))
        return K.reshape(c, shape=(-1, 1, latent_dim))

    apply_attention_vector = Lambda(apply_attention, output_shape=(1, latent_dim), name='apply_attention')

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(3000, activation='softmax', name='decoder_dense')
    concat_layer = Concatenate(axis=-1)

    encoder_outputs = Input(shape=(80, latent_dim))
    decoder_inputs = Input(shape=(1, 3000))
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))

    transformed_state_h = attention_dense_h(decoder_state_input_h)
    repeated_transformed_state_h = repeat_vector(transformed_state_h)
    transformed_encoder_outputs = attention_dense_e(encoder_outputs)

    attention_raw = add_layer([repeated_transformed_state_h, transformed_encoder_outputs])
    attention_pre = tanh_activate(attention_raw)
    attention_vector = softmax_activate(flatten_layer(attention_dense(attention_pre)))

    attention_context = apply_attention_vector([attention_vector, encoder_outputs])
    fused_input = concat_layer([decoder_inputs, attention_context])
    decoder_outputs, state_h, state_c = decoder_lstm(fused_input,
                                                     initial_state=[decoder_state_input_h, decoder_state_input_c])
    dense_output = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs, encoder_outputs, decoder_state_input_h, decoder_state_input_c],
                          [dense_output, state_h, state_c, attention_vector])

    decoder_model.load_weights('model_final', by_name=True)
    # decoder_model.get_layer(name='attention_dense').set_weights(np.load('attention_dense.npy'))
    # decoder_model.get_layer(name='decoder_lstm').set_weights(np.load('decoder_lstm.npy'))
    # decoder_model.get_layer(name='decoder_dense').set_weights(np.load('decoder_dense.npy'))

    return encoder_model, decoder_model


def inference_sample_beam_search(sample, tokenizer_word_index, encoder_model, decoder_model, log_file=None):
    BOS_token = tokenizer_word_index['bos']
    EOS_token = tokenizer_word_index['eos']

    reverse_word_map = dict(map(reversed, tokenizer_word_index.items()))
    reverse_word_map[0] = '0'
    reverse_word_map[BOS_token] = 'BOS'
    reverse_word_map[EOS_token] = 'EOS'

    encoder_outputs, state_h, state_c = encoder_model.predict(sample)

    beam_search_n = 10
    sentence = [[0]] * beam_search_n
    states = [[]] * beam_search_n
    scores = np.zeros([beam_search_n])
    # scores.fill('-inf')

    states_temp = [[]] * beam_search_n
    scores_temp = np.zeros([beam_search_n, beam_search_n])
    token_temp = np.zeros([beam_search_n, beam_search_n], dtype=np.int)

    sentence[0] = [BOS_token]
    states[0] = [state_h, state_c]
    scores[0] = 1.

    while True:
        scores_temp.fill(0.)
        token_temp.fill(0)
        skip = 0

        for n in range(beam_search_n):

            input_token = sentence[n][-1]

            if input_token == 0:
                scores_temp[n, 0] = scores[n]
                skip += 1
                continue

            input_onehot = np_utils.to_categorical(input_token, num_classes=3000).reshape([1, 1, -1])
            output, state_h, state_c, attention = decoder_model.predict([input_onehot, encoder_outputs] + states[n])
            states_temp[n] = [state_h, state_c]

            tokens = (-output[0, 0, :]).argsort()[:beam_search_n]
            probabilities = output[0, 0, tokens]

            token_temp[n] = tokens
            scores_temp[n] = probabilities * scores[n]

            print('at %dth beam: (beam probability: %f)' % (n, scores[n]), file=log_file)
            print(' '.join([reverse_word_map[i] for i in sentence[n]]), file=log_file)
            print('    next word: P(word|beam), P(beam+word)', file=log_file)

            for i in range(beam_search_n):
                print('        %s: %f, %f' % (reverse_word_map[tokens[i]], probabilities[i], scores_temp[n, i]), file=log_file)

        if skip == beam_search_n:
            break

        # print(scores_temp)

        top_score = (-scores_temp.flatten()).argsort()[:beam_search_n]

        print('Score top %d:' % beam_search_n, file=log_file)

        last_sentence = list(sentence)
        for n in range(beam_search_n):
            source = int(top_score[n] / beam_search_n)
            element = int(top_score[n] % beam_search_n)
            sentence[n] = last_sentence[source] + [token_temp[source, element]]
            states[n] = states_temp[source]
            scores[n] = scores_temp[source, element]
            print('    (%d, %d)(%f): %s' % (source, element, scores[n],
                                            ' '.join([reverse_word_map[i] for i in sentence[n]])), file=log_file)

    reverse_word_map[0] = ''
    reverse_word_map[BOS_token] = ''
    reverse_word_map[EOS_token] = ''

    scaled_score = np.zeros([beam_search_n])
    for i in range(beam_search_n):
        l = 0
        while sentence[i][l] != 0:
            l += 1
        scaled_score[i] = np.log10(scores[i]) / l

    # print(scaled_score)

    final_sentence = ' '.join([reverse_word_map[i] for i in sentence[scaled_score.argmax()]])
    final_sentence = final_sentence.strip(' ')

    print('\nFinal sentence:%s' % final_sentence, file=log_file)

    return final_sentence


def inference_sample(sample, tokenizer, encoder_model, decoder_model):
    BOS_token = tokenizer.word_index['bos']
    EOS_token = tokenizer.word_index['eos']
    target = np_utils.to_categorical(BOS_token, num_classes=3000).reshape([1, 1, -1])

    encoder_outputs, state_h, state_c = encoder_model.predict(sample)

    decoded_sentence = []
    attention_list = []
    while True:
        output, state_h, state_c, attention = decoder_model.predict([target, encoder_outputs, state_h, state_c])
        token = np.argmax(output[0, -1, :])

        # target = np_utils.to_categorical(token, num_classes=3000).reshape([1, 1, -1])
        target = output

        decoded_sentence += [token]
        attention_list += [attention]

        if token == EOS_token or len(decoded_sentence) > 50:
            break

    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    reverse_word_map[0] = ''
    reverse_word_map[EOS_token] = '.'

    return ' '.join([reverse_word_map[i] for i in decoded_sentence]), np.vstack(attention_list)


latent_dim = 512
# data_dir = '../MLDS_hw2_data/'
try:
    data_dir = sys.argv[1]
    test_output_filename = sys.argv[2]
    peer_output_filename = sys.argv[3]
except IndexError:
    print('Usage: python hw2_seq2seq_final.py [data directory] [test result output file] [peer result output file]')
    exit(0)

if mode == 'train':
    train_id_list, train_data_np, train_label_list, tokenizer = load_train_data(data_dir)
    test_id_list, test_data_np, test_label_list = load_test_data(data_dir, tokenizer)
    train_data_gen = data_generator(train_data_np, train_label_list, batch_size)
    test_data_gen = data_generator(test_data_np, test_label_list, 100)

    training_model = make_train_model(latent_dim)
    train_model(training_model, train_data_gen, test_data_gen, epochs=800)
elif mode == 'continue':
    train_id_list, train_data_np, train_label_list, tokenizer = load_train_data(data_dir)
    test_id_list, test_data_np, test_label_list = load_test_data(data_dir, tokenizer)
    train_data_gen = data_generator(train_data_np, train_label_list, batch_size)
    test_data_gen = data_generator(test_data_np, test_label_list, 100)

    training_model = make_train_model(latent_dim, adam_clip=0.01)
    training_model.load_weights('model', by_name=True)
    train_model(training_model, train_data_gen, test_data_gen, epochs=100)
elif mode == 'predict':
    tokenizer_word_index = json.load(open('tokenizer_word_index.json', 'r', encoding='utf-8'))
    _mean = np.load('_mean.npy')
    _std = np.load('_std.npy')

    encoder_model, decoder_model = load_inference_models(latent_dim)

    all_lines = open(data_dir + 'testing_id.txt', 'r', encoding='utf-8').readlines()
    with open(test_output_filename, 'w', encoding='utf-8') as f:
        for filename in all_lines:
            filename = filename.strip(' \n')
            data = np.load(data_dir + 'testing_data/feat/' + filename + '.npy')
            data = ((data - _mean) / (2 * _std)).reshape([-1, 80, 4096])
            with open('logs/' + filename + '.log', 'w', encoding='utf-8') as log_f:
                result = inference_sample_beam_search(data, tokenizer_word_index, encoder_model, decoder_model, log_file=log_f)
            # result, attention_mat = inference_sample(data, tokenizer, encoder_model, decoder_model)
            f.write(filename + ',' + result + '\n')

    all_lines = open(data_dir + 'peer_review_id.txt', 'r', encoding='utf-8').readlines()
    with open(peer_output_filename, 'w', encoding='utf-8') as f:
        for filename in all_lines:
            filename = filename.strip(' \n')
            data = np.load(data_dir + 'peer_review/feat/' + filename + '.npy')
            data = ((data - _mean) / (2 * _std)).reshape([-1, 80, 4096])
            with open('logs/' + filename + '.log', 'w', encoding='utf-8') as log_f:
                result = inference_sample_beam_search(data, tokenizer_word_index, encoder_model, decoder_model, log_file=log_f)
            # result, attention_mat = inference_sample(data, tokenizer, encoder_model, decoder_model)
            f.write(filename + ',' + result + '\n')
