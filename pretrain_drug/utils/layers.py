import tensorflow as tf
from inits import glorot


conv1d = tf.layers.conv1d
        
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):   #参数后面有等号，传递时顺序可以不一样；seq是特征矩阵
    with tf.name_scope('my_attn'):
        #generate linear features
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  #8个滤波器，宽度为1，长度默认为特征长度。每个滤波器产生一个2708x1的向量，最终seq_fts为2708x8 h'
        #seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
        
        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) #1546x1
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) #1546x1
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])   #e_ij   1546x1546计算过程类似于两不同维度矩阵相乘 （1）
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)   #每一行权值之和为1
        
        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
            
        vals = tf.matmul(coefs, seq_fts)     #neighbor representation aggreggation
        ret = tf.contrib.layers.bias_add(vals)   #h'  为最后的向量 （4）（激活前）

        return activation(ret), coefs[0]  # activation  为最后的向量，对应公式（4）

def sp_attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0):   #参数后面有等号，传递时顺序可以不一样；seq是特征矩阵
    with tf.name_scope('my_attn'):
        
        #seq_fts = seq
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)
            
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  #8个滤波器，宽度为1，长度默认为特征长度。每个滤波器产生一个2708x1的向量，最终seq_fts为2708x8 h'    
        
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
    
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])   #6375x6375
        seq_fts = tf.squeeze(seq_fts)  #6375x64
        ret = tf.sparse.sparse_dense_matmul(coefs, seq_fts)  #first-order neighbours    张量 1000x64

        return activation(ret)  # activation  为最后的向量，对应公式（4）  

       
    