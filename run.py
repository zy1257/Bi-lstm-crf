import tensorflow as tf
import numpy as np
import pickle
from data_process import read_batch, tag_label, vocab, id_to_vocab, label_tag
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import viterbi_decode


class BiLSTM_CRF():
    def __init__(self):
        self.lr = 0.001
        self.dropout = 1.0
        self.vocab_len = 4658
        self.num_tags = 7
        self.embedding_dim = 300
        self.batch_size = 64
        self.epoch = 20
        self.train_path = '.\\data\\train_data'
        self.test_path = '.\\data\\test_data'
    def build_graph(self):
        self.y_data = tf.placeholder(dtype = 'int32', shape=[None, None],name = 'y_data')
        self.x_data = tf.placeholder(dtype = 'int32', shape=[None, None],name = 'x_data')
        self.seq_len = tf.placeholder(dtype = 'int32', shape=[None],name = 'seq_len')

        embedding = tf.get_variable('embedding',[self.vocab_len, self.embedding_dim])
        # (batch_size, max_len, embedding_dim)
        self.word_embedding = tf.nn.embedding_lookup(embedding,self.x_data,name='word_embedding')
        
        with tf.variable_scope('BiLSTM'):
            cell_fw = LSTMCell(self.embedding_dim)
            cell_bw = LSTMCell(self.embedding_dim)
            (output_fw_seq, output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                               cell_bw=cell_bw,
                                                                               inputs=self.word_embedding,
                                                                               sequence_length=self.seq_len,
                                                                               dtype = tf.float32)
            # (batch_size, max_len, 2*embedding_dim)
            out_put = tf.concat([output_fw_seq,output_bw_seq],axis=-1)
            max_len = tf.shape(out_put)[1]
            # (batch_size*max_len, 2*embedding_dim)
            out_put = tf.reshape(out_put,[-1, 2*self.embedding_dim])
        with tf.variable_scope('proj'):
            # (batch_size*max_len, num_tags)
            fc = tf.layers.dense(out_put, self.num_tags)
            fc = tf.contrib.layers.dropout(fc,self.dropout)
            self.logits = tf.reshape(fc,[self.batch_size, max_len, self.num_tags])
        with tf.variable_scope('crf'):
            log_likelihood, self.transition = crf_log_likelihood(inputs = self.logits,
                                                                 tag_indices = self.y_data,
                                                                 sequence_lengths = self.seq_len)
            self.loss = -tf.reduce_mean(log_likelihood)
        
        optim = tf.train.AdamOptimizer(self.lr)
        self.train_op = optim.minimize(self.loss)

        tf.summary.scalar("loss", self.loss)
        self.merge = tf.summary.merge_all()

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('.\\tensorboard',sess.graph)
            
            for epoch in range(self.epoch):
                data_train = read_batch(self.train_path,self.batch_size,vocab,tag_label)
                for step,((x_train,y_train), seq_len_list) in enumerate(data_train):
                    merged,_ = sess.run([self.merge,self.train_op],feed_dict=
                                        {self.x_data:x_train,self.y_data:y_train,self.seq_len:seq_len_list})
                    writer.add_summary(merged,step)
                saver.save(sess, '.\\model/BLO.model{}'.format(epoch))
    def test(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess,'.\\model/BLO.model18')
            
            data_test = read_batch(self.test_path,self.batch_size,vocab,tag_label)
            with open('.\\result/test_pre18.txt','w',encoding='utf-8') as f:
                for (x_test,y_test), seq_len_list in data_test:
                    logits, transition_params = sess.run([self.logits,self.transition],feed_dict=
                                                         {self.x_data:x_test,self.y_data:y_test,self.seq_len:seq_len_list})
                    label_list,write_all = [],[]
                    for logit, seq_len in zip(logits, seq_len_list):
                        viterbi_seq,_ = viterbi_decode(logit[:seq_len],transition_params)
                        label_list.append(viterbi_seq)
                    for x_write,y_write,y_pre,num in zip(x_test,y_test,label_list,seq_len_list):
                        for i in range(num):
                            write_temp = ''
                            write_temp += id_to_vocab[x_write[i]] + ' ' + label_tag[y_write[i]] + ' ' + label_tag[y_pre[i]] + '\n'
                            f.write(write_temp)
            
model = BiLSTM_CRF()
model.build_graph()
model.train()
model.test()
