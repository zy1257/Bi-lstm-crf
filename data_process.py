import pickle
from tensorflow.contrib import keras

def read_batch(data_dir, batch_size, vocab, tag_label):
    with open(data_dir,encoding='utf-8') as f:
        data,sent,labels,sent_pad,labels_pad,seq_len = [],[],[],[],[],[]
        for i in f:
            if i != '\n':
                char, label = i.strip().split(sep='\t')
                if char not in vocab:
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
                
def demo_data(s):
    x_input = []
    for i in s:
        if i in vocab:
            x_input.append(vocab[i])
        else:
            x_input.append(vocab['UNK'])
    return [x_input], [len(x_input)]

def demo_output(s,t):
    L = len(s)
    B,plo = [1,3,5],''
    PER, LOC, ORG = [],[],[]
    for index, label in enumerate(t):
        if label in B:
            plo += s[index]
            if index < L:
                j = index + 1
                while t[j]:
                    plo += s[j]
                    j += 1
                    if j == L:
                        break
            if label == 1:
                PER.append(plo)
            elif label == 3:
                LOC.append(plo)
            elif label == 4:
                ORG.append(plo)
            plo = ''
    print('人名：',PER,'\n','地名：',LOC,'\n','机构名：',ORG, sep='')

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open('label_to_vocab.pkl','rb') as f:
    label_to_vocab = pickle.load(f)
with open('embedding.pkl','rb') as f:
    embedding = pickle.load(f)
tag_label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6}
label_tag = {0: 'O',
             1: 'B-PER', 2: 'I-PER',
             3: 'B-LOC', 4: 'I-LOC',
             5: 'B-ORG', 6: 'I-ORG'}

