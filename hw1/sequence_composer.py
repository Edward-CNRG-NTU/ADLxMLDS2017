from data_processing import *
import time


def compose_sequence_by_sample(id, result, phone_full_dict):
    if result.shape[0] != id.shape[0]:
        print(id.shape, result.shape)
        print('length mismatch!')
        return None

    sample_dict = {}
    for i in range(id.shape[0]):
        composed_id = id[i].decode('utf8')
        character = phone_full_dict[result[i]]['mapped_to']['char']
        split_id = composed_id.split('_')
        sample_name = split_id[0] + '_' + split_id[1]
        sample_index = int(split_id[2])

        try:
            if sample_index == 1:
                sample_dict[sample_name] = {}
                sample_dict[sample_name]['sequence'] = character
                sample_dict[sample_name]['reduced'] = character
                sample_dict[sample_name]['count'] = 1
            else:
                if sample_dict[sample_name]['count'] != (sample_index - 1):
                    print(len(sample_dict[sample_name]), sample_index, 'out of order!')
                    return None
                else:
                    sample_dict[sample_name]['count'] += 1

                sample_dict[sample_name]['sequence'] += character

                if sample_dict[sample_name]['reduced'][-1] != character:
                    sample_dict[sample_name]['reduced'] += character

        except KeyError as e:
            print('KeyError', e)
            return None

    for key in sample_dict.keys():
        sample_dict[key]['reduced'] = sample_dict[key]['reduced'].strip('L')

    return sample_dict


if __name__ == '__main__':
    phone_full_dict = load_phone_dict()
    # fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_text(phone_full_dict)
    fbank_test_id, fbank_test_data, fbank_train_id, fbank_train_data, mfcc_test_id, mfcc_test_data, mfcc_train_id, mfcc_train_data, train_label = load_from_binary()
    t1 = time.time()
    sample_dict = compose_sequence_by_sample(fbank_train_id, train_label, phone_full_dict)
    print(time.time() - t1)
    print(sample_dict.keys())
    print(sample_dict['faem0_si1392'])

    with open('trainletter.csv', 'r') as f:
        for line in f.readlines():
            id, seq = line.strip('\n').split(',')
            try:
                if sample_dict[id]['reduced'] != seq:
                    print(sample_dict[id]['reduced'], seq)
                else:
                    print('ok')
            except KeyError:
                print('unknown id:', id)
