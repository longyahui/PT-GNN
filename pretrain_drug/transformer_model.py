
from models.gat import GAT, GAT_new
from models.gcn import GraphConvolutionSparse, GraphConvolution
from models.mgcn import MolecularGraphConvolution
from models.bert import BERT
import tensorflow as tf
import numpy as np
from metrics import masked_accuracy, masked_accuracy_batch
import scipy.sparse as sp
from inits import sparse_to_tuple

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
    
class TransformerModel: 
    def __init__(self, Model, encoded_drugs, mol_graphs, do_train=True):
       self.batch_size = 1 
       self.lr = 0.005  
       self.l2_coef = 0.0005  
       self.weight_decay = 5e-4
       self.token_size = 43
       self.dim_embedding = 100
       self.len_sequence = 600
       self.Model = Model
       self.mol_graph = mol_graphs
       self.encoded_drug = encoded_drugs
       
       self.moModel = MolecularGraphConvolution(
                           input_dim = self.dim_embedding,
                           output_dim = 128,
                           act=tf.nn.leaky_relu)
       self.weight_decay = 1e-4
       self.num_filter = 1
       if do_train:
           self.num_nodes = 3543 
       else:
           self.num_nodes = 1971
       self.entry_size = self.num_nodes**2    
           
       with tf.name_scope('input'):
         self.bias_in = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph')
         self.lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
         self.msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
         self.neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
         self.attn_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())
         self.ffd_drop = tf.compat.v1.placeholder(dtype=tf.float32, shape=())    
       
       self.instantiate_embeddings()
       
       self.logits = self.inference()
       self.loss, self.accuracy = self.loss_func()
       #self.accuracy = self.loss
       self.train_op = self.train()
    
       
    def inference(self):
        emb_drugs = self.embedding_drugs()  #transform token sequences to protein
        emb_drugs = tf.nn.l2_normalize(emb_drugs, 1)   #[3543, 128]
           
        #embedding_proteins = self.embedding_protein
           
        self.model = GraphConvolutionSparse(
                        input_dim = emb_drugs.shape[1].value,
                        output_dim = 512,
                        adj= self.bias_in,
                        act=tf.nn.leaky_relu,
                        dropout = 0.25)
        self.final_embedding = self.model.encoder(emb_drugs)
        logits = self.model.decoder(self.final_embedding, self.num_nodes)
        return logits
    
    def embedding_drugs(self):
        ids = list(self.mol_graph.keys())
        emb_drugs = []
        #for modular_id in ids:
        for i in range(len(ids)):
            adj = self.mol_graph[ids[i]]
            drug_sequence = self.encoded_drug[i]
            inputs = self.get_features(drug_sequence)
            
            x = self.moModel.encoder(inputs, adj)   #对分子图做卷积操作   [21, 128]
            x = x[tf.newaxis]                #[1, 21, 128]
            x = tf.layers.max_pooling1d(x, pool_size=x.shape[1].value, strides=x.shape[1].value)  #[1,1,128]
            emb_drugs.append(tf.contrib.layers.flatten(x))
        emb_drugs = tf.concat(emb_drugs, axis=0)   #[3543, 128]
        return emb_drugs
    
    def get_features(self, sequence):
        sequence = np.array(sequence).reshape([1,len(sequence)])
        sequence = tf.cast(sequence, dtype=tf.int32)
        embedding = tf.nn.embedding_lookup(self.embedding_tokens, sequence) #[1,21,100]
        return embedding[0]
        
        
    def loss_func(self):
        loss = masked_accuracy(self.logits, self.lbl_in, self.msk_in, self.neg_msk)
        #para_decode = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="deco")
        #loss +=  self.weight_decay * tf.nn.l2_loss(para_decode)
        #accuracy = masked_accuracy(self.logits, self.lbl_in1, self.msk_in1, self.neg_msk1)
        accuracy = loss
        return loss, accuracy
    
    def train(self):
        train_op = self.model.training(self.loss, self.lr, self.l2_coef)
        return train_op
    
    def instantiate_embeddings(self):
        """define all embeddings here"""
        with tf.name_scope("token_embedding"):  # embedding matrix
           self.embedding_tokens = tf.get_variable("embedding", shape=[self.token_size, self.dim_embedding],initializer=tf.random_normal_initializer(stddev=0.1))     
    
    def SimpleAttLayer(self, inputs, attention_size, time_major=False, return_alphas=False):   #inputs.size=(6375,3,64)
        hidden_size = inputs.shape[2].value   #64

        # Trainable parameters
        w_omega = tf.Variable(tf.random.normal([hidden_size, attention_size], stddev=0.1))    #attention_size=64 w_omega=[64,32]
        b_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random.normal([attention_size], stddev=0.1))  #u_omega=[32,]

        with tf.name_scope('v'):
           v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)  #(6375,3,32)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape   (6375,3)
    
        alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape  (6375,3)    
    
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)   #(6375,64,1)
        #print("output:", output)
        return output        