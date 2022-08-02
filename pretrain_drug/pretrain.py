import os
import time
import numpy as np
import scipy.io as sio
import tensorflow as tf

from transformer_model import TransformerModel
from inits import adj_to_bias, preprocess_graph
from inits import test_negative_sample
from inits import sparse_to_tuple
from inits import train_negative_sample
from metrics import masked_accuracy
from metrics import ROC
from dataloader import load_data_DDI

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
epochs = 200  
for _ in range(1):
    interaction, y_train, train_mask, labels, encoded_drugs, mol_graphs = load_data_DDI()
    biases = preprocess_graph(interaction)
    
    save_path = "saved_model/model.ckpt"
    model = TransformerModel(encoded_drugs, mol_graphs)
    saver = tf.train.Saver()
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    with tf.compat.v1.Session() as sess: 
       sess.run(init_op)
       train_loss_avg = 0
       train_acc_avg = 0
       for epoch in range(epochs):
           t = time.time()
           
           ##########    train     ##############
           neg_mask = train_negative_sample(labels, len(labels))
           _, loss_value_tr, acc_tr, emb, scores = sess.run([model.train_op, model.loss, model.accuracy, model.embedding_tokens, model.logits],
                             feed_dict={
                                 model.bias_in: biases,                               
                                 model.lbl_in: y_train,                                
                                 model.msk_in: train_mask,                               
                                 model.neg_msk: neg_mask})
           print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr, acc_tr, time.time()-t))
       saver.save(sess, save_path)
       sess.close()    

       
