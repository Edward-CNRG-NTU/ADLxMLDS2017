import pickle
import json
import numpy as np
from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

hair_tags = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
             'green hair', 'red hair', 'purple hair', 'pink hair',
             'blue hair', 'black hair', 'brown hair', 'blonde hair']

eyes_tags = ['gray eyes', 'black eyes', 'orange eyes',
             'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
             'green eyes', 'brown eyes', 'red eyes', 'blue eyes']

num_classes = len(hair_tags) + len(eyes_tags) + 1

tokenizer = Tokenizer(num_words=num_classes,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~\t\n',
                      lower=True, split=',', char_level=False)

tokenizer.fit_on_texts(hair_tags + eyes_tags)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(tokenizer.word_index)
print(tokenizer.word_counts)

y_tags = []
with open('tags_clean.csv', 'r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        n, tags = line.split(',')

        if int(n) != len(y_tags):
            print('index error at line %d: %s' % (len(y_tags), line))
            exit(0)
        else:
            text = ','.join([tag.split(':')[0].rstrip(' ') for tag in tags.rstrip('\t').split('\t')])
            y_tags.append(text)

# json.dump(y_tags, open('y_tags_onehot.txt', 'w'), separators=(',\n', ':'))


def tags_to_onehot(tags):
    seqs = tokenizer.texts_to_sequences(tags)
    return np.array([np.sum(to_categorical(seq, num_classes), axis=-2) for seq in seqs])

y_onehot = tags_to_onehot(y_tags)

print(y_onehot[:20])

np.save('y_onehot.npy', y_onehot)

np.save('hair_onehot.npy', tags_to_onehot(hair_tags))
np.save('eyes_onehot.npy', tags_to_onehot(eyes_tags))

eyes_hair_tags = ['%s,%s' % (eyes_tag, hair_tag) for eyes_tag in eyes_tags for hair_tag in hair_tags]
eyes_hair_onehot = tags_to_onehot(eyes_hair_tags)

np.save('eyes_hair_onehot.npy', eyes_hair_onehot)
