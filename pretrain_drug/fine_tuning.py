from inits import div_list
import os
import time
import tensorflow as tf
import numpy as np
from inits import preprocess_graph
from transformer_model import TransformerModel
from inits import test_negative_sample, train_negative_sample
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from metrics import ROC
from dataloader import load_data_for_fine_tuning, load_drug_graph

epochs = 100
batch_size = 1
num_nodes = 1971
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def train(train_arr, test_arr, model):
    interaction, y_train, y_test, train_mask, test_mask, labels = load_data_for_fine_tuning(train_arr, test_arr)
    biases = preprocess_graph(interaction)
    
    init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint("./save_model/"))
        train_loss_avg = 0
        train_acc_avg = 0
        neg_mask = train_negative_sample(labels, len(train_arr), do_train=False)
        
        for epoch in range(epochs):
          t = time.time()
          ##########    train     ##############
          tr_step = 0
          tr_size = 1 
          while tr_step * batch_size < tr_size:  
              _, loss_value_tr, acc_tr = sess.run([model.train_op, model.loss, model.accuracy],
                      feed_dict={
                           model.bias_in: biases,
                           model.lbl_in: y_train,
                           model.msk_in: train_mask,
                           model.neg_msk: neg_mask})
              train_loss_avg += loss_value_tr
              train_acc_avg += acc_tr
              tr_step += 1   
          print('Epoch: %04d | Training: loss = %.5f, acc = %.5f, time = %.5f' % ((epoch+1), loss_value_tr,acc_tr, time.time()-t))
        print("Finish training.")
          
        ###########     test      ############
        ts_size = 1
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0
    
        print("Start to test")
        while ts_step * batch_size < ts_size:
              out_come, loss_value_ts, acc_ts = sess.run([model.logits, model.loss, model.accuracy],
                      feed_dict={
                          model.bias_in: biases,
                          model.lbl_in: y_test,
                          model.msk_in: test_mask,
                          model.neg_msk: neg_mask})
              ts_loss += loss_value_ts
              ts_acc += acc_ts
              ts_step += 1
        print('Test loss:', ts_loss/ts_step, '; Test accuracy:', ts_acc/ts_step)
              
        out_come = out_come.reshape((num_nodes,num_nodes))
        test_negative_samples = test_negative_sample(labels,len(test_arr),neg_mask.reshape((num_nodes,num_nodes)))
        test_labels, score = ROC(out_come,labels, test_arr,test_negative_samples)  
              
        return test_labels, score
        sess.close() 
          
          
# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

labels = np.loadtxt("data/adj_ddi_fine_tuning.txt")  
reorder = np.arange(labels.shape[0])
np.random.shuffle(reorder)

encoded_drugs, mol_graph = load_drug_graph()

T = 1
cv_num = 5

label, score = [], []

for t in range(T):
    #print("t: %d" % t)
    order = div_list(reorder.tolist(),cv_num)
    aupr_vec=[]
    auc_vec=[]
    model = TransformerModel(encoded_drugs, mol_graph, do_train=False)
    for i in range(cv_num):
        test_arr = order[i]
        arr = list(set(reorder).difference(set(test_arr)))
        np.random.shuffle(arr)
        train_arr = arr
    
        test_labels, scores = train(train_arr, test_arr, model)  
        
        prec, rec, thr = precision_recall_curve(test_labels, scores)
        aupr_val = auc(rec, prec)
        aupr_vec.append(aupr_val)
        fpr, tpr, thr = roc_curve(test_labels,scores)
        auc_val = auc(fpr, tpr)
        auc_vec.append(auc_val)
        
        print ("auc:%.6f, aupr:%.6f" % (auc_val, aupr_val))
        
        label.append(test_labels)
        score.append(scores)
        
        plt.figure
        plt.plot(fpr,tpr)
        plt.show()
        plt.figure
        plt.plot(rec,prec)
        plt.show()
        
    aupr_mean=np.mean(aupr_vec)
    aupr_std=np.std(aupr_vec) 
    auc_mean=np.mean(auc_vec)
    auc_std=np.std(auc_vec)
    
    print ("auc_ave:%.6f, auc_std: %.6f, aupr_ave:%.6f, aupr_std:%.6f" % (auc_mean, auc_std, aupr_mean, aupr_std))
    
    
   
    
    
   
       
