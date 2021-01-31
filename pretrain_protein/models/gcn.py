import tensorflow as tf
import numpy as np

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""
    def __init__(self, n_node, input_dim, output_dim, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = "Convolution"
        self.vars = {}
        self.issparse = False
        self.n_node = n_node
        with tf.compat.v1.variable_scope(self.name):
            self.vars['weights1'] = glorot([input_dim, output_dim])
            self.vars['weights2'] = weight_variable_glorot(output_dim, 64, name='weights2')
            
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.norm = norm
        self.is_train = is_train

    def encoder(self, inputs, adj):
        with tf.name_scope(self.name):
             x = inputs
             x = tf.matmul(x, self.vars['weights1'])
             x = tf.sparse_tensor_dense_matmul(adj, x)
             outputs = self.act(x)
               
             x2  =  tf.matmul(outputs, self.vars['weights2'])
             x2 = tf.sparse_tensor_dense_matmul(adj, x2)
             outputs = self.act(x2)

        if self.norm:
           outputs = tf.layers.batch_normalization(outputs, training=self.is_train)
        return outputs
    
    def decoder(self, embed, nd):
        embed_size = embed.shape[1].value
        logits=tf.matmul(embed,tf.transpose(embed))
        logits=tf.reshape(logits,[-1,1])
        #return logits
        return tf.nn.relu(logits)
    
    def training(self, loss, lr, l2_coef):
        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        
        # training op
        train_op = opt.minimize(loss)
        return train_op