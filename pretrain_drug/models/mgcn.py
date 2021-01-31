import tensorflow as tf
import numpy as np
import scipy.sparse as sp

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim], minval=-init_range,
        maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class MolecularGraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, norm=False):
        self.name = 'moleculargraph'
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.act = act
        self.norm = norm

    def encoder(self, inputs, adj):
        with tf.name_scope(self.name):
            adj = preprocess_adj(adj)
            adj = tf.cast(adj, tf.float32)
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.matmul(adj, x)
            outputs = self.act(x)
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs)

        return outputs