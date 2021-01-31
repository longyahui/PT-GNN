import tensorflow as tf
import numpy as np


def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask  
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))

def ROC(outs, labels, test_arr, label_neg):
    scores=[]
    for i in range(len(test_arr)):
        l=test_arr[i]
        scores.append(outs[int(labels[l, 0]-1),int(labels[l, 1]-1)])
        
    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]),int(label_neg[i, 1])])
        
    test_labels=np.ones((len(test_arr), 1))
    temp=np.zeros((label_neg.shape[0], 1))
    test_labels1=np.vstack((test_labels, temp))
    test_labels1=np.array(test_labels1, dtype=np.bool).reshape([-1,1])
    return test_labels1, scores