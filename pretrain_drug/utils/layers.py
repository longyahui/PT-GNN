import tensorflow as tf
from inits import glorot


conv1d = tf.layers.conv1d
        
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):  
    with tf.name_scope('my_attn'):
        
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  
        #seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
 
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) 
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])   
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)   
        
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            
        vals = tf.matmul(coefs, seq_fts)     
        ret = tf.contrib.layers.bias_add(vals)  

        return activation(ret), coefs[0]  

def sp_attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):   
    with tf.name_scope('my_attn'):
        
        #seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
            
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)     
        
        nb_nodes = seq_fts.shape[1].value
        
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) #1546x1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) #1546x1
        
        logits = tf.add(f_1[0], tf.transpose(f_2[0]))
        logits_first = bias_mat * logits
    
        lrelu = tf.SparseTensor(indices=logits_first.indices,
                                values=tf.nn.leaky_relu(logits_first.values),
                                dense_shape=logits_first.dense_shape)
        coefs = tf.sparse_softmax(lrelu)
    
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])  
        seq_fts = tf.squeeze(seq_fts) 
        ret = tf.sparse.sparse_dense_matmul(coefs, seq_fts)  

        return activation(ret) 

       
    
