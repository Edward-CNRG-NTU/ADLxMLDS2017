import numpy as np


def normalize(data, axis=0):
    _mean = np.mean(data, axis=axis)
    _std = np.std(data, axis=axis)
    return (data - _mean) / _std, _mean, _std


def load_phone_dict(data_dir):
    phone_48_to_char_raw = np.loadtxt(data_dir + '/48phone_char.map', delimiter='\t', dtype=np.string_)
    phone_full_dict = {}
    for phone in phone_48_to_char_raw:
        phone_str = phone[0].decode('utf8')
        char_str = phone[2].decode('utf8')
        phone_full_dict[phone_str] = {'phone': phone_str, 'phone_id': int(phone[1]), 'char': char_str}

    phone_48_to_39_raw = np.loadtxt(data_dir + '/phones/48_39.map', delimiter='\t', dtype=np.string_)
    for phone in phone_48_to_39_raw:
        phone_str = phone[0].decode('utf8')
        phone_full_dict[phone_str]['mapped_to'] = phone_full_dict[phone[1].decode('utf8')]
        # phone_48_full_dict[phone_str]['mapped_char'] = phone_48_full_dict[phone[1]]['char']
        # phone_48_full_dict[phone_str]['mapped_phone_id'] = phone_48_full_dict[phone[1]]['char']
        phone_full_dict[phone_full_dict[phone_str]['phone_id']] = phone_full_dict[phone_str]

    print('load_phone_dict done!')
    return phone_full_dict


def load_from_text(data_dir):
    try:
        fbank_test_id = np.loadtxt(data_dir + '/fbank/test.ark', usecols=0, dtype=np.str_)
        fbank_test_data = np.loadtxt(data_dir + '/fbank/test.ark', usecols=(range(1, 70)))
        print('load /fbank/test.ark done!')
        fbank_train_id = np.loadtxt(data_dir + '/fbank/train.ark', usecols=0, dtype=np.str_)
        fbank_train_data = np.loadtxt(data_dir + '/fbank/train.ark', usecols=(range(1, 70)))
        print('load /fbank/train.ark done!')
        mfcc_test_id = np.loadtxt(data_dir + '/mfcc/test.ark', usecols=0, dtype=np.str_)
        mfcc_test_data = np.loadtxt(data_dir + '/mfcc/test.ark', usecols=(range(1, 40)))
        print('load /mfcc/test.ark done!')
        mfcc_train_id = np.loadtxt(data_dir + '/mfcc/train.ark', usecols=0, dtype=np.str_)
        mfcc_train_data = np.loadtxt(data_dir + '/mfcc/train.ark', usecols=(range(1, 40)))
        print('load /mfcc/train.ark done!')
    except IOError:
        print('some files may not exist.')
        exit(1)
    except ValueError:
        print('binary files may have corrupted.')
        exit(1)

    fbank_train_data, fbank_mean, fbank_std = normalize(fbank_train_data)
    fbank_test_data = (fbank_test_data - fbank_mean) / fbank_std

    mfcc_train_data, mfcc_mean, mfcc_std = normalize(mfcc_train_data)
    mfcc_test_data = (mfcc_test_data - mfcc_mean) / mfcc_std

    # try:
    #     fbank_test_id = np.load('fbank_test_id.npy')
    #     fbank_test_data = np.load('fbank_test_data.npy')
    #     print('load /fbank/test.ark done!')
    #     fbank_train_id = np.load('fbank_train_id.npy')
    #     fbank_train_data = np.load('fbank_train_data.npy')
    #     print('load /fbank/train.ark done!')
    #     mfcc_test_id = np.load('mfcc_test_id.npy')
    #     mfcc_test_data = np.load('mfcc_test_data.npy')
    #     print('load /mfcc/test.ark done!')
    #     mfcc_train_id = np.load('mfcc_train_id.npy')
    #     mfcc_train_data = np.load('mfcc_train_data.npy')
    #     print('load /mfcc/train.ark done!')
    # except IOError:
    #     print('some files may not exist. load from text...')
    #
    #     try:
    #         fbank_test_id = np.loadtxt(data_dir + '/fbank/test.ark', usecols=0, dtype=np.str_)
    #         fbank_test_data = np.loadtxt(data_dir + '/fbank/test.ark', usecols=(range(1, 70)))
    #         print('load /fbank/test.ark done!')
    #         fbank_train_id = np.loadtxt(data_dir + '/fbank/train.ark', usecols=0, dtype=np.str_)
    #         fbank_train_data = np.loadtxt(data_dir + '/fbank/train.ark', usecols=(range(1, 70)))
    #         print('load /fbank/train.ark done!')
    #         mfcc_test_id = np.loadtxt(data_dir + '/mfcc/test.ark', usecols=0, dtype=np.str_)
    #         mfcc_test_data = np.loadtxt(data_dir + '/mfcc/test.ark', usecols=(range(1, 40)))
    #         print('load /mfcc/test.ark done!')
    #         mfcc_train_id = np.loadtxt(data_dir + '/mfcc/train.ark', usecols=0, dtype=np.str_)
    #         mfcc_train_data = np.loadtxt(data_dir + '/mfcc/train.ark', usecols=(range(1, 40)))
    #         print('load /mfcc/train.ark done!')
    #     except IOError:
    #         print('some files may not exist.')
    #         exit(1)
    #     except ValueError:
    #         print('binary files may have corrupted.')
    #         exit(1)
    #
    #     fbank_train_data, fbank_mean, fbank_std = normalize(fbank_train_data)
    #     fbank_test_data = (fbank_test_data - fbank_mean) / fbank_std
    #
    #     mfcc_train_data, mfcc_mean, mfcc_std = normalize(mfcc_train_data)
    #     mfcc_test_data = (mfcc_test_data - mfcc_mean) / mfcc_std
    #
    #     np.save('fbank_test_id', fbank_test_id)
    #     np.save('fbank_test_data', fbank_test_id)
    #     np.save('fbank_train_id', fbank_test_id)
    #     np.save('fbank_train_data', fbank_test_id)
    #     np.save('mfcc_test_id', fbank_test_id)
    #     np.save('mfcc_test_data', fbank_test_id)
    #     np.save('mfcc_train_id', fbank_test_id)
    #     np.save('mfcc_train_data', fbank_test_id)

    return fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, \
           mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data


def prepare_label_from_text(data_dir, phone_full_dict, fbank_train_id, mfcc_train_id):
    train_label_raw = np.loadtxt(data_dir + '/label/train.lab', delimiter=',', dtype=np.str_)

    # print(train_label_raw[:5], train_label_raw.shape)
    train_label_dict = {}
    for pair in train_label_raw:
        train_label_dict[pair[0]] = pair[1]

    train_label = np.zeros(train_label_raw.shape[0], dtype=np.int8)
    for i in range(train_label_raw.shape[0]):
        if fbank_train_id[i] == mfcc_train_id[i]:
            train_label[i] = phone_full_dict[train_label_dict[fbank_train_id[i]]]['phone_id']
            # print(fbank_train_id[i], train_label_dict[fbank_train_id[i]], train_label_in_order[i])
        else:
            print('not match at %d' % i)
            exit(1)

    print('load /label/train.lab done!')
    return train_label


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


def export_result(sample_dict, model_dir='', file_name='result.csv', output_raw=False):
    with open('sample.csv', 'r') as fr:
        sample_format = fr.readlines()

    with open(model_dir + file_name, 'w') as fw:
        fw.write(sample_format[0])
        for line in sample_format[1:]:
            id = line.split(',')[0]
            try:
                fw.write(id + ',' + sample_dict[id]['reduced'] + '\n')
            except KeyError:
                print('id: %s not found!' % id)

    if output_raw:
        with open(model_dir + file_name + '_raw.csv', 'w') as fw:
            fw.write(sample_format[0])
            for line in sample_format[1:]:
                id = line.split(',')[0]
                try:
                    fw.write(id + ',' + sample_dict[id]['raw'] + '\n')
                except KeyError:
                    print('id: %s not found!' % id)
