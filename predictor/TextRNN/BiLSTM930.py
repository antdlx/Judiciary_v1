import json
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import rnn
import math
from os.path import join

from mjudger import Judger

SEQ_LEN = 200
err = 100032

def get_accu_dic(acc_path):
    '''
    生成罪名-id的字典,注意，比赛要求是从1开始，但是后面函数中的labels要求从0开始，所以后面别忘了+1
    :param acc_path:
    :return:
    '''
    accu_dic = {}
    counter = 0
    with open(acc_path,"r",encoding="utf8") as file:
        for line in file:
            accu_dic[line[:-1]] = counter
            counter += 1
    return accu_dic

def get_word2id(path):
    '''
    获得Word2id的映射关系
    :param path:
    :return:
    '''
    with open(path,'r',encoding='utf-8',errors=None) as f:
        word2id = eval(f.read())
    return word2id


def get_data(fact_path,label_path):
    '''
    生成fact和label
    :param fact_path:
    :param label_path:
    :return:
    '''
    facts = []
    labels = []
    fact_counter = 0
    err_counter = 0
    with open(fact_path,"r",encoding="utf8") as file:
        for line in file:
            fact_counter += 1
            facts.append(line.split())
            if fact_counter==err:
                print("FACT {0}".format(line.split()))

    with open(label_path,"r",encoding="utf8") as file:
        for line in file:
            err_counter += 1
            if err_counter == err:
                print("ERROR {0}".format(json.loads(line)['meta']['accusation']))
                continue
            labels.append(json.loads(line)['meta']['accusation'])
    return facts,labels


def id_and_pad(facts,labels,word2id,label_dic):
    '''
    将传入的数据转换成id的形式
    :param facts: 二维数组格式，第二维度是词，如[[公诉 机关 起诉 指控 被告人 张 某某 秘密 窃取 财物 价 值],[...]...]
    :param labels: 二维数组格式，第二维度是词，如[[盗窃，抢劫],[...],...]
    :param word2id:
    :param label_dic:
    :return:facts_id是[[2,4,25,56,...],[...],...]
            labels_id是[[190,16],[190],...]
            facts_len是[62,45,...]，表示每句话的实际长度
    '''
    facts_id = []
    facts_len = []
    labels_id = []
    for line in facts:
        line_id = []
        for word in line:
            if word in word2id:
                line_id.append(word2id[word])

            if len(line_id) == SEQ_LEN:
                break
        facts_len.append(len(line_id))
        while len(line_id) < SEQ_LEN:
            line_id.append(0)
        facts_id.append(line_id)
    for line in range(len(labels)):
        tmp = []
        # line_id = []
        #暂时只取第一个
        # for accusation in line:
        #     if accusation in label_dic:
        #         line_id.append(label_dic[accusation])
        #     else:
        #         line_id.append(0)
        if labels[line][0] in label_dic:
            tmp.append(label_dic[labels[line][0]])
        else:
            tmp.append(0)
        tmp.append(facts_len[line])
        labels_id.append(tmp)

    facts_id = np.asarray(facts_id)
    labels_id = np.asarray(labels_id)
    return facts_id,labels_id

# def weight(shape, stddev=0.1, mean=0):
#     initial = tf.truncated_normal(shape=shape, mean=mean, stddev=stddev)
#     return tf.Variable(initial)
#
# def bias(shape, value=0.1):
#     initial = tf.constant(value=value, shape=shape)
#     return tf.Variable(initial)

def main():
    train_batch_size = 128
    dev_batch_size = 128
    test_batch_size = 64
    embedding_size = 128
    hidden_size = 64
    dropout_prob = 0.5
    learning_rate = 0.01
    decay_steps=1000
    decay_rate=0.9
    global_step = tf.Variable(-1, trainable=False, name="Global_Step")
    summaries_dir = 'summaries/'
    epoch_num = 3
    steps_per_print = 100
    steps_per_summary = 2
    epochs_per_dev = 1
    epochs_per_save = 1
    checkpoint_dir = "ckpt/model.ckpt"

    acc = get_accu_dic('../../datas/accu.txt')
    word2id = get_word2id('../../datas/dic/word2id.txt')

    vocab_size = len(word2id) + 1
    num_classes = len(acc)

    train_facts, train_labels = get_data('../../datas/cutted/data_train.txt', '../../datas/data_train.json')
    valid_facts, valid_labels = get_data('../../datas/cutted/data_valid.txt', '../../datas/data_valid.json')
    test_facts, test_labels = get_data('../../datas/cutted/data_test.txt', '../../datas/data_test.json')
    train_fact_ids, train_label_ids = id_and_pad(train_facts, train_labels, word2id, acc)
    valid_fact_ids, valid_label_ids = id_and_pad(valid_facts, valid_labels, word2id, acc)
    test_fact_ids, test_label_ids = id_and_pad(test_facts, test_labels, word2id, acc)

    # Steps
    train_steps = math.ceil(len(train_fact_ids) / train_batch_size)
    dev_steps = math.ceil(len(valid_fact_ids) / dev_batch_size)
    test_steps = math.ceil(len(test_fact_ids) / test_batch_size)

    # Train and dev dataset
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_fact_ids, train_label_ids))
    train_dataset = train_dataset.shuffle(10000).batch(train_batch_size)

    dev_dataset = tf.contrib.data.Dataset.from_tensor_slices((valid_fact_ids, valid_label_ids))
    dev_dataset = dev_dataset.shuffle(10000).batch(dev_batch_size)

    test_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_fact_ids, test_label_ids))
    test_dataset = test_dataset.batch(test_batch_size)

    # A reinitializable iterator
    #tf.int32 tf.int32      [None,1000]
    iterator = tf.contrib.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

    train_initializer = iterator.make_initializer(train_dataset)
    dev_initializer = iterator.make_initializer(dev_dataset)
    test_initializer = iterator.make_initializer(test_dataset)

    # Input Layer
    with tf.variable_scope('inputs'):
        x, y_fix = iterator.get_next()
        y_label = y_fix[:,0]
        x_len = y_fix[:,1]

    with tf.name_scope("embedding"):
        embedding_table = np.loadtxt("../../datas/dic/embedding_table")
        embedding = tf.get_variable('embedding',[vocab_size,embedding_size],initializer=tf.constant_initializer(embedding_table))

        # W_projection = tf.get_variable("W_projection", shape=[hidden_size * 2, num_classes],
        #                                initializer=tf.random_normal_initializer(stddev=0.1))  # [embed_size,label_size]
        # b_projection = tf.get_variable("b_projection", shape=[num_classes])  # [label_size]

    # 1.get emebedding of words in the sentence
    embedded_words = tf.nn.embedding_lookup(embedding, x)

    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    #2. Bi-lstm layer
    # define lstm cess:get lstm cell output
    # lstm_fw_cell=rnn.BasicLSTMCell(hidden_size) #forward direction cell
    # lstm_bw_cell=rnn.BasicLSTMCell(hidden_size) #backward direction cell
    gru_fw_cell = rnn.GRUCell(hidden_size)
    gru_bw_cell = rnn.GRUCell(hidden_size)
    # lstm_fw_cell = rnn.LSTMCell(hidden_size)
    # lstm_bw_cell = rnn.LSTMCell(hidden_size)
    if dropout_keep_prob is not None:
        # lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout_keep_prob)
        # lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout_keep_prob)
        gru_fw_cell = rnn.DropoutWrapper(gru_fw_cell, output_keep_prob=dropout_keep_prob)
        gru_bw_cell = rnn.DropoutWrapper(gru_bw_cell, output_keep_prob=dropout_keep_prob)
    #???加入sequence_length节省计算资源
    # outputs,_=tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,inputs=embedded_words,dtype=tf.float32) #[batch_size,sequence_length,hidden_size]

    outputs, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell, inputs=embedded_words, dtype=tf.float32,sequence_length=x_len)
    print("outputs:===>",outputs)
    #3. concat output
    output_rnn=tf.concat(outputs,axis=2) #[batch_size,sequence_length,hidden_size*2]
    output_rnn_last=output_rnn[:,-1,:] ##[batch_size,hidden_size*2]
    print("output_rnn_last:", output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>
    #4. logits(use linear layer)
    with tf.name_scope("output"): #inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward activations of the input network.
        w = tf.get_variable("W_projection", shape=[hidden_size * 2, num_classes],
                                            initializer=tf.random_normal_initializer(stddev=0.1))  # [embed_size,label_size]
        b = tf.get_variable("b_projection", shape=[num_classes])
        # w = weight([hidden_size * 2, num_classes])
        # b = bias([num_classes])
        logits = tf.matmul(output_rnn_last, w) + b  # [batch_size,num_classes]

    with tf.name_scope("loss"):
        y_label_reshape = tf.cast(tf.reshape(y_label, [-1]), tf.int32)
        l2_lambda = 0.0001
        # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
        # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_reshape,
                                                                logits=logits)  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
        print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
        loss = tf.reduce_mean(losses)  # print("2.loss.loss:", loss) #shape=()
        tv = tf.trainable_variables()
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses

    # """based on the loss, use SGD to update parameter"""
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate,
                                               staircase=True)

    train_op = tf.contrib.layers.optimize_loss(loss, global_step=global_step, learning_rate=learning_rate,
                                               optimizer="Adam")

    predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
    correct_prediction = tf.equal(predictions, y_label_reshape) #tf.argmax(self.logits, 1)-->[batch_size]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
    print("loss:",loss,"acc:",accuracy,"label:",y_label,"prediction:",predictions)

    # Saver
    saver = tf.train.Saver()

    # Iterator
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    gstep = 0

    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(join(summaries_dir, 'train'),
                                   sess.graph)

    # if tf.gfile.Exists(summaries_dir):
    #     tf.gfile.DeleteRecursively(summaries_dir)

    for epoch in range(epoch_num):
        tf.train.global_step(sess, global_step_tensor=global_step)
        # Train
        sess.run(train_initializer)
        for step in range(int(train_steps)):
            smrs, loss_, acc_, gstep, _ = sess.run([summaries, loss, accuracy, global_step, train_op],
                                                     feed_dict={dropout_keep_prob:dropout_prob})
            # Print log
            if step % steps_per_print == 0:
                print('Global Step', gstep, 'Step', step, 'Train Loss', loss_, 'Accuracy', acc_)

            # Summaries for tensorboard
            if gstep % steps_per_summary == 0:
                writer.add_summary(smrs, gstep)
                # print('Write summaries to', summaries_dir)

        if epoch % epochs_per_dev == 0:
            # Dev
            sess.run(dev_initializer)
            for step in range(int(dev_steps)):
                if step % steps_per_print == 0:
                    print('Dev Accuracy', sess.run(accuracy,
                                                     feed_dict={dropout_keep_prob:1}), 'Step', step)

        # Save model
        if epoch % epochs_per_save == 0:
            if os.path.exists('../outputs_BiLSTM/data_test.json'):
                with open('../outputs_BiLSTM/data_test.json', "r+", encoding='utf8') as f:
                    f.truncate()
            sess.run(test_initializer)
            for step in range(int(test_steps)):
                if step % steps_per_print == 0:
                    print("testing:{}".format(step))
                predict_test = sess.run(predictions,feed_dict={dropout_keep_prob:1})
                # print("PT:-------------------")
                # print(predict_test.tolist())
                # print(x_)
                result = ''
                for l in range(len(predict_test)):
                    ans = {}
                    # print(predict_test[l])
                    tmp = predict_test.tolist()
                    tmp2 = tmp[l]
                    tmp3 = []
                    tmp3.append(tmp2)
                    ans['accusation'] = tmp3
                    ans['articles'] = [0]
                    ans['imprisonment'] = 0
                    result += json.dumps(ans)+"\n"
                with open('../outputs_BiLSTM/data_test.json',"a+",encoding='utf8') as f:
                    f.write(result)

            saver.save(sess, checkpoint_dir, global_step=gstep)

            judger = Judger("accu.txt", "law.txt")
            res = judger.test("../datas", "../outputs_BiLSTM")
            score = judger.get_score(res)
            ouf = open(("predict_log"), "a+",encoding='utf8')
            print(score,file=ouf)
            ouf.close()


    # ckpt = tf.train.get_checkpoint_state('../ckpt')
    # saver.restore(sess, ckpt.model_checkpoint_path)
    # print('Restore from', ckpt.model_checkpoint_path)
    # sess.run(test_initializer)
    # for step in range(int(test_steps)):
    #     y = sess.run(predictions, feed_dict={dropout_keep_prob: 1})
    #     # print(_)
    #     # print("------------------")
    #     print(y)


if __name__ == '__main__':
    main()