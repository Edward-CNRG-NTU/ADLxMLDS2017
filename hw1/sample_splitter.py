from keras.utils import np_utils
from data_processing import *
import time


def get_sample_info_by_name(id, max_len=1000):
    sample_info_list= []
    sample_info = None
    for i in range(id.shape[0]):
        composed_id = id[i]
        split_id = composed_id.split('_')
        sample_name = split_id[0] + '_' + split_id[1]
        sample_index = int(split_id[2])

        if (sample_index % max_len) == 1:
            if sample_info is None:
                pass
            else:
                sample_info[2] = i - 1
                sample_info[3] = i - sample_info[1]
                sample_info_list.append(sample_info)
            sample_info = ['', 0, 0, 0]
            sample_info[0] = sample_name
            sample_info[1] = i
        else:
            pass
    i += 1
    sample_info[2] = i - 1
    sample_info[3] = i - sample_info[1]
    sample_info_list.append(sample_info)

    return sample_info_list


def parse_sample_by_name(sample_id, max_sequence):
    sample_info_list = get_sample_info_by_name(sample_id, max_sequence)
    sample_indexer = get_sample_indexer_by_sample_info(sample_info_list, max_sequence)
    return sample_indexer, sample_info_list


def get_sample_indexer_by_sample_info(sample_info_list, max_len=1000):
    indexer = np.empty((len(sample_info_list), max_len), dtype=np.int32)
    indexer.fill(-1)
    for i in range(len(sample_info_list)):
        sample_info = sample_info_list[i]
        indexer[i, :sample_info[3]] = np.arange(sample_info[1], sample_info[2] + 1)

    return indexer


def make_sliding_indexer(source_length, window_size, stride=1):
    return np.arange(window_size)[None, :] + np.arange(0, source_length - (window_size - 1), stride)[:, None]


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


if __name__ == '__main__':
    phone_full_dict = load_phone_dict()
    # fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_text(phone_full_dict)
    fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_binary()
    t1 = time.time()
    sample_info_list = get_sample_info_by_name(mfcc_train_id, 800)
    sample_indexer = get_sample_indexer_by_sample_info(sample_info_list, 800)
    print(time.time() - t1)
    print(sample_info_list[:5])
    print(sample_indexer[:5])