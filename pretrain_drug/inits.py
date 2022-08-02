import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
import networkx as nx
from rdkit import Chem

def adj_to_bias(adj, sizes, nhood=1):       
    nb_graphs = adj.shape[0] 
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)                            
            
def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()
    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    
def train_negative_sample(labels, N, do_train=True):  
    num = 0
    if do_train:
      num_node = 3543 
    else:
      num_node = 1971
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(num_node, num_node)).toarray()
    A = A + A.T
    mask = np.zeros(A.shape)
    while(num<2*N):
        a = random.randint(0,num_node-1) 
        b = random.randint(0,num_node-1) 
        if  A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            num += 1           
    mask = np.reshape(mask,[-1,1])  
    return mask

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    (num_node, _)=negative_mask.shape
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(num_node, num_node)).toarray()  
    A = A + A.T
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,num_node-1) 
        b = random.randint(0,num_node-1) 
        if a<b and A[a,b] != 1 and mask[a,b] != 1 and negative_mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1    
    return test_neg

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def maxpooling(a):
    a=tf.cast(a,dtype=tf.float32)
    b=tf.reduce_max(a,axis=1,keepdims=True)
    c=tf.equal(a,b)
    mask=tf.cast(c,dtype=tf.float32)
    final=tf.multiply(a,mask)
    ones=tf.ones_like(a)
    zeros=tf.zeros_like(a)
    final=tf.where(final>0.0,ones,zeros)
    return final

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized
    
def sparse_matrix(matrix):
    sigma = 0.001
    matrix = matrix.astype(np.int32)
    result = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        if matrix[i,0]==0:
           result[i,0]=sigma
        else:
           result[i,0]=1
    return result

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape  

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)
    
