import thulac
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.externals import joblib
import re
import os


def get_word2id(path):
    '''
    获得Word2id的映射关系
    :param path:
    :return:
    '''
    with open(path, 'r', encoding='utf-8', errors=None) as f:
        word2id = eval(f.read())
    return word2id

class Predictor(object):
    def __init__(self):
        # self.tfidf = joblib.load('predictor/model_jieba/tfidf.model')
        # self.law = joblib.load('predictor/model_jieba/law.model')
        # self.accu = joblib.load('predictor/model_jieba/accu.model')
        # self.time = joblib.load('predictor/model_jieba/time.model')
        self.word2id = get_word2id("predictor/res/word2id.txt")
        hidden_size = 64
        num_classes = 202
        embedding_size = 128
        seq_len = 1000
        vocab_size = len(self.word2id)+1

        with tf.name_scope("embedding"):
            embedding_table = np.loadtxt("predictor/res/embedding_table")
            embedding = tf.get_variable('embedding', [vocab_size, embedding_size],
                                        initializer=tf.constant_initializer(embedding_table))
        self.vec = tf.placeholder(tf.int32, shape=[1, seq_len], name="vec")
        embedded_words = tf.nn.embedding_lookup(embedding, self.vec)

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # forward direction cell
        lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # backward direction cell
        if self.dropout_keep_prob is not None:
            lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        # ???加入sequence_length节省计算资源
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words,
                                                     dtype=tf.float32)  # [batch_size,sequence_length,hidden_size]
        print("outputs:===>", outputs)
        # 3. concat output
        output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        output_rnn_last = output_rnn[:, -1, :]  ##[batch_size,hidden_size*2]
        print("output_rnn_last:", output_rnn_last)  # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        # 4. logits(use linear layer)
        with tf.name_scope(
                "output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
            w = self.weight([hidden_size * 2, num_classes])
            b = self.bias([num_classes])
            logits = tf.matmul(output_rnn_last, w) + b  # [batch_size,num_classes]

        self.predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)

        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state('predictor/TextRNN/ckpt')
        saver.restore(self.sess, ckpt.model_checkpoint_path)


        self.batch_size = 1

        self.cut = thulac.thulac(seg_only=True)

    def predict_law(self, vec):
        # y = self.law.predict(vec)
        # return [y[0] + 1]
        return [0]

    def weight(self,shape, stddev=0.1, mean=0):
        initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
        return tf.Variable(initial)

    def bias(self,shape, value=0.1):
        initial = tf.constant(value=value, shape=shape)
        return tf.Variable(initial)

    def predict_accu(self, word_vec):

        # hidden_size = 64
        # num_classes = 202
        # embedding_size = 128
        # seq_len = 1000
        #
        # with tf.name_scope("embedding"):
        #     embedding_table = np.loadtxt("predictor/res/embedding_table")
        #     embedding = tf.get_variable('embedding', [vocab_size, embedding_size],
        #                                 initializer=tf.constant_initializer(embedding_table))
        # embedded_words = tf.nn.embedding_lookup(embedding, word_vec)
        #
        # dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)  # forward direction cell
        # lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)  # backward direction cell
        # if dropout_keep_prob is not None:
        #     lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
        #     lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
        # # ???加入sequence_length节省计算资源
        # outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, embedded_words,
        #                                              dtype=tf.float32)  # [batch_size,sequence_length,hidden_size]
        # print("outputs:===>", outputs)
        # # 3. concat output
        # output_rnn = tf.concat(outputs, axis=2)  # [batch_size,sequence_length,hidden_size*2]
        # output_rnn_last = output_rnn[:, -1, :]  ##[batch_size,hidden_size*2]
        # print("output_rnn_last:", output_rnn_last)  # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
        # # 4. logits(use linear layer)
        # with tf.name_scope(
        #         "output"):  # inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        #     w = self.weight([hidden_size * 2, num_classes])
        #     b = self.bias([num_classes])
        #     logits = tf.matmul(output_rnn_last, w) + b  # [batch_size,num_classes]
        #
        # predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        #
        # saver = tf.train.Saver()
        # sess = tf.Session()
        # sess.run(tf.global_variables_initializer())
        #
        # ckpt = tf.train.get_checkpoint_state('predictor/TextRNN/ckpt')
        # saver.restore(sess, ckpt.model_checkpoint_path)

        y = self.sess.run(self.predictions, feed_dict={self.dropout_keep_prob: 1,self.vec:word_vec})

        return [y[0] + 1]

    def predict_time(self, vec):

        # y = self.time.predict(vec)[0]
        y = 0

        # 返回每一个罪名区间的中位数
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def id_and_pad(self,facts,word2id,SEQ_LEN=1000):
        '''
        将传入的数据转换成id的形式
        :param facts: 二维数组格式，第二维度是词，如[[公诉 机关 起诉 指控 被告人 张 某某 秘密 窃取 财物 价 值],[...]...]
        :param word2id:
        :return:facts_id是[[2,4,25,56,...],[...],...]
        '''
        facts_id = []
        for word in facts:
            if word in word2id:
                facts_id.append(word2id[word])
            if len(facts_id) == SEQ_LEN:
                break
        while len(facts_id) < SEQ_LEN:
            facts_id.append(0)
        res = []
        res.append(facts_id)

        res = np.asarray(res)
        return res

    def clean_detected_words(self,cutted_list):
        """
        清除停用词
        :param cutted_list:
        :return:
        """
        detected_list = []

        with open("predictor/res/Detected_words.txt",'r',encoding='utf-8',errors='ignore') as f:
            detected_list = f.readlines()

        #下面2行代码的含义是删除最后的\n符号
        for i in range(0,len(detected_list)):
            detected_list[i] = detected_list[i][:-1]

        detected_dics = {}.fromkeys(detected_list,1)
        result = []
        for cutted in cutted_list:
            if cutted not in detected_dics:
                result.append(cutted)
        return result

    def predict(self, content):

        #clean
        after_clean = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！【】，×X。？：、~@#￥%……&*（）；]+|[A-Z]+|[a-z]", "", content[0])
        after_clean_cut =self.cut.cut(after_clean, text=True)
        after_clean_cut = after_clean_cut.split()
        after_clean_cut_detected = self.clean_detected_words(after_clean_cut)
        vec = self.id_and_pad(after_clean_cut_detected,self.word2id)

        ans = {}

        ans['accusation'] = self.predict_accu(vec)
        ans['articles'] = self.predict_law(vec)
        ans['imprisonment'] = self.predict_time(vec)

        print(ans)
        return [ans]


