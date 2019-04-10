# bi-lstm-crf<br>
## 目的<br>
识别3种命名实体，人名、地名、以及有关部门名<br>
对比随机生成嵌入层词向量和使用预训练的词向量效果差别<br>
命名实体使用如下label{O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG}，其中B为一个词的开头，I为非开头，O对应其他<br>
## 数据<br>
所使用数据可到此处下载 https://pan.baidu.com/s/1RYdjzqkLcQkPCPiu9R0tyg 提取码: kw44<br>
预训练词向量可到[Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)下载，本文使用的是人民日报的word 300d<br>
## 模型<br>
### 嵌入层 embedding<br>
    embedding = tf.Variable(self.embedding,dtype = tf.float32)
    self.word_embedding = tf.nn.embedding_lookup(embedding,self.x_data,name='word_embedding')
### bi-lstm层<br>
    cell_fw = LSTMCell(300)
    cell_bw = LSTMCell(300)
    (output_fw_seq, output_bw_seq),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                     cell_bw=cell_bw,
                                                                     inputs=self.word_embedding,
                                                                     sequence_length=self.seq_len,
                                                                     dtype = tf.float32)
预训练的测向量维度为300，这里cell_num统一为300<br>
### 全连接层<br>
    fc = tf.layers.dense(out_put, self.num_tags)
    fc = tf.contrib.layers.dropout(fc,self.dropout)
    self.logits = tf.reshape(fc,[self.batch_size, max_len, self.num_tags])
### crf层<br>
    log_likelihood, self.transition = crf_log_likelihood(inputs = self.logits,
                                                         tag_indices = self.y_data,
                                                         sequence_lengths = self.seq_len)
    self.loss = -tf.reduce_mean(log_likelihood)
## 模型效果对比<br>
### 不使用预训练词向量<br>
    processed 171959 tokens with 6174 phrases; found: 5923 phrases; correct: 4996.
    accuracy:  97.82%; precision:  84.35%; recall:  80.92%; FB1:  82.60
                  LOC: precision:  89.69%; recall:  84.73%; FB1:  87.14  2716
                  ORG: precision:  78.47%; recall:  79.41%; FB1:  78.94  1347
                  PER: precision:  80.81%; recall:  76.37%; FB1:  78.53  1860
从测试集来看，是有一定效果的，但只是对人名的识别效果较好<br>
### 使用预训练的词向量<br>
    processed 171959 tokens with 6174 phrases; found: 6051 phrases; correct: 5297.
    accuracy:  98.33%; precision:  87.54%; recall:  85.80%; FB1:  86.66
                  LOC: precision:  90.58%; recall:  88.00%; FB1:  89.27  2793
                  ORG: precision:  82.71%; recall:  82.64%; FB1:  82.68  1330
                  PER: precision:  86.46%; recall:  84.71%; FB1:  85.57  1928
使用预训练词向量后，对三种命名实体的识别都有一定的提升，后续如果希望继续提升效果，可以考虑预训练词向量。                  
