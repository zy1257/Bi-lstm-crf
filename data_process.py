import pickle
from tensorflow.contrib import keras

def read_batch(data_dir, batch_size, vocab, tag_label):
    with open(data_dir,encoding='utf-8') as f:
        data,sent,labels,sent_pad,labels_pad,seq_len = [],[],[],[],[],[]
        for i in f:
            if i != '\n':
                char, label = i.strip().split(sep='\t')
                if char.isdigit():
                    sent.append(vocab['NUM'])
                elif ('\u0041' <= char <= '\u005a') or ('\u0061' <= char <= '\u007a'):
                    sent.append(vocab['ENG'])
                elif char not in vocab:
                    sent.append(vocab['UNK'])
                else:
                    sent.append(vocab[char])
                labels.append(tag_label[label])
            else:
                sent_pad.append(sent)
                labels_pad.append(labels)
                seq_len.append(len(sent))
                sent,labels = [],[]
            if len(sent_pad) == batch_size:
                sent_pad = keras.preprocessing.sequence.pad_sequences(sent_pad,padding = 'post')
                labels_pad = keras.preprocessing.sequence.pad_sequences(labels_pad,padding = 'post')
                data.append(sent_pad)
                data.append(labels_pad)
                yield data, seq_len
                data,sent_pad,labels_pad,seq_len = [],[],[],[]

def build_vocab():
    with open('.\\data\\train_data',encoding='utf-8') as f:
        vocab = {}
        for i in f:
            if i != '\n':
                char,_ = i.strip().split(sep='\t')
            if char <= '\u9fa5' and char >= '\u4e00':
                if char not in vocab:
                    vocab[char] = 1
        L = vocab.keys()
        for value, key in enumerate(L):
            vocab[key] = value
        vocab['NUM'] = len(L)
        vocab['ENG'] = len(L) + 1
        vocab['UNK'] = len(L) + 2
    id_to_vocab = {}
    for key, value in vocab.items():
        id_to_vocab[value] = key
    with open('vocab.pkl','wb') as f:
        pickle.dump(vocab,f)
    with open('id_to_vocab.pkl','wb') as f:
        pickle.dump(id_to_vocab,f)

build_vocab()

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('id_to_vocab.pkl','rb') as f:
    id_to_vocab = pickle.load(f)
tag_label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6}
label_tag = {0: 'O',
             1: 'B-PER', 2: 'I-PER',
             3: 'B-LOC', 4: 'I-LOC',
             5: 'B-ORG', 6: 'I-ORG'}
