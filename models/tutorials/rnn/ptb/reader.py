from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import re

import tensorflow as tf

Py3 = sys.version_info[0] == 3
tag = {'D':0,'O': 1, 'B': 2, 'I': 3, 'E': 4, 'S': 5}
seq_size = 30

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read()
        else:
            return f.read().decode("utf-8")

def _build_vocab(all_str):
    counter = collections.Counter(all_str)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(1, len(words)+1)))

    return word_to_id

def data_to_word_ids(raw_data, word_to_id):
    all_seq_to_id = []
    for seq in raw_data:
        seq_to_id = [word_to_id[word] for word in seq[0] if word in word_to_id]
        l = len(seq_to_id)   
        while(l < seq_size):
            seq_to_id.append(0)
            l += 1
        all_seq_to_id.append([seq_to_id, seq[1], seq[2]])
    return all_seq_to_id  
        
# get valid data
def data_pre_processing(data):
    sentn = re.findall(r'\n*([\s\S]*?)\s+\w+_set\s+\d+\s+\d+.*', data)
    data_split = data.split('\n')
    raw_data = []
    all_str = ''
    j = 0
    for sp in data_split:
        index = re.findall(r'\s+\w+_set\s+(\d+)\s+(\d+)', sp)
        l = len(index)
        in_str = ''
        tag_num = [0] * seq_size
        tag_weight = []
        count = 0
        flag = True
        if(l == 1):
            sen_l = len(sentn[j])
            s = int(index[0][0])
            e = int(index[0][1])
            if(sen_l >= e and s >= 0 and sentn[j][s] != ' ' and sentn[j][e-1] != ' ' ):
                for i in range(sen_l):
                    word = sentn[j][i]
                    if(word == ' '): 
                        count += 1
                    elif(word == '\n'):
                        continue
                    else: 
                        in_str += word
                    if(i == s): s -= count
                    if(i == e-1): e -= count
                if(len(in_str) > seq_size):
                    j += 1 
                    continue
                for i in range(len(in_str)):
                    if(s == e and i == s):
                        tag_num[i] = tag['S']
                    elif(s < e and i == s):
                        tag_num[i] = tag['B'] 
                    elif(s < e and i > s and i < e):
                        tag_num[i] = tag['I']     
                    elif(s < e and i == e):
                        tag_num[i] = tag['E']
                    else:
                        tag_num[i] = tag['O']
                tag_weight = len(in_str) 
                all_str += in_str
                raw_data.append([in_str, tag_num, tag_weight])
            j += 1
        elif(l == 2):
            sen_l = len(sentn[j])
            s1 = int(index[0][0])
            e1 = int(index[0][1])
            s2 = int(index[1][0])
            e2 = int(index[1][1])
            if(sen_l >= e1 and s1 >= 0 and sentn[j][s1] != ' ' and sentn[j][e1-1] != ' ' and\
               sen_l >= e2 and s2 >= 0 and sentn[j][s2] != ' ' and sentn[j][e2-1] != ' '):
                for i in range(sen_l):
                    word = sentn[j][i]
                    if(word == ' '):
                        count += 1
                    elif(word == '\n'):
                        continue
                    else:
                        in_str += word
                    if(i == s1): s1 -= count
                    if(i == s2): s2 -= count                    
                    if(i == e1-1): e1 -= count
                    if(i == e2-1): e2 -= count
                if(len(in_str) > seq_size):
                    j += 1
                    continue
                for i in range(len(in_str)): 
                    if((s1 == e1 and i == s1) or (s2 == e2 and i == s2)) :
                        tag_num[i] = tag['S']
                    elif((s1 < e1 and i == s1) or (s2 < e2 and i == s2)):
                        tag_num[i] = tag['B']
                    elif((s1 < e1 and i > s1 and i < e1) or (s2 < e2 and i > s2 and i < e2)):
                        tag_num[i] = tag['I']
                    elif((s1 < e1 and i == e1) or (s2 < e2 and i == e2)): 
                        tag_num[i] = tag['E']
                    else:
                        tag_num[i] = tag['O']
                tag_weight = len(in_str)
                all_str += in_str
                raw_data.append([in_str, tag_num, tag_weight])
            j += 1
    word_to_id = _build_vocab(all_str)
    raw_data = data_to_word_ids(raw_data, word_to_id)
    return raw_data

# read files & processing data
def ptb_raw_data(data_path=None):
    train_all_data = ''
    valid_all_data = ''
    test_all_data = ''
    for i in range(1):
        if(i == 0):
            train_path = os.path.join(data_path, "tagging_train0.txt")
        elif(i == 1):
            train_path = os.path.join(data_path, "tagging_train1.txt")
        elif(i == 2):
            train_path = os.path.join(data_path, "tagging_train2.txt")
        elif(i == 3):
            train_path = os.path.join(data_path, "tagging_train3.txt")
        elif(i == 4):
            train_path = os.path.join(data_path, "tagging_train4.txt")
        else:
            train_path = os.path.join(data_path, "tagging_train5.txt")
        train_all_data += _read_words(train_path)
    for i in range(1):
        if(i == 0):
            valid_path = os.path.join(data_path, "tagging_valid0.txt")
        elif(i == 1):
            valid_path = os.path.join(data_path, "tagging_valid1.txt")
        else:
            valid__path = os.path.join(data_path, "tagging_valid2.txt")            
        valid_all_data += _read_words(valid_path)
    test_path = os.path.join(data_path, "tagging_test.txt")
    test_all_data = _read_words(test_path)
    
    train_data = data_pre_processing(train_all_data)
    valid_data = data_pre_processing(valid_all_data)
    test_data = data_pre_processing(test_all_data)
    return train_data, valid_data, test_data

# Returns:Tensors, x, y shaped (batch, steps), z shaped (batch)
def ptb_producer(raw_data, batch_size, num_steps, name=None):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        input_data = []
        target_data = []
        weight_data = []
        
        for single_data in raw_data:
            input_data.append(single_data[0])
            target_data.append(single_data[1])
            weight_data.append(single_data[2])
        input_data = tf.convert_to_tensor(input_data, name="input_data", dtype=tf.int32)
        target_data = tf.convert_to_tensor(target_data, name="target_data", dtype=tf.int32)
        weight_data = tf.convert_to_tensor(weight_data, name="weight_data", dtype=tf.int32)
        batch_len = tf.size(input_data)
        epoch_size = batch_len // batch_size // num_steps


        assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
       
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(input_data, [i * batch_size, 0],[(i+1) * batch_size, num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(target_data, [i * batch_size, 0],[(i+1) * batch_size, num_steps])
        y.set_shape([batch_size, num_steps]) 
        z = tf.strided_slice(weight_data, [i * batch_size],[(i+1) * batch_size]) 
        z.set_shape([batch_size]) 
        return x, y, z
 
