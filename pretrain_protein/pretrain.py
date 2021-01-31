import os
import time
import numpy as np
import tensorflow as tf

from model import Model
from inits import preprocess_graph
from inits import load_data
from inits import train_negative_sample
from metrics import masked_accuracy
from metrics import ROC

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
epochs = 200
def train():
    interaction1, interaction2, logits_train1, logits_train2, train_mask1, train_mask2, labels1, labels2, word_matrix = load_data()
    biases1 = preprocess_graph(interaction1)
    biases2 = preprocess_graph(interaction2)
    save_path = "saved_model/model.ckpt"
    model = Model()
    saver = tf.train.Saver()
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess: 
       sess.run(init_op)
       neg_mask1, neg_mask2 = train_negative_sample(logits_train1, logits_train2, len(labels1), len(labels2))
       for epoch in range(epochs):
           t = time.time()  
           _, loss_value_tr, acc_tr, emb = sess.run([model.train_op, model.loss, model.accuracy, model.embedding_tokens],
                    feed_dict={
                        model.encoded_protein: word_matrix,
                        model.bias_in1: biases1, 
                        model.bias_in2: biases2,                              
                        model.lbl_in1: logits_train1,
                        model.lbl_in2: logits_train2,                                
                        model.msk_in1: train_mask1,
                        model.msk_in2: train_mask2,                               
                        model.neg_msk1: neg_mask1,
                        model.neg_msk2: neg_mask2})
           print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr, acc_tr, time.time()-t))
       saver.save(sess, save_path)
       sess.close()

train() 

       
