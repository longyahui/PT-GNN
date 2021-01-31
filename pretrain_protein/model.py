from models.gcn import GraphConvolutionSparse
import tensorflow as tf
from metrics import masked_accuracy

class Model: 
    def __init__(self, do_train=True):
       self.batch_size = 1 
       self.batch_num = 64
       self.lr = 0.005  
       self.l2_coef = 0.0005  
       self.weight_decay = 5e-3
       self.nonlinearity = tf.nn.relu
       self.token_size = 8001
       self.dim_embedding = 100
       self.len_sequence = 600
       self.num_filter = 1
       self.do_train = do_train
       if self.do_train:
           self.num_nodes = 20375 
       else:
           self.num_nodes = 6375
       self.entry_size = self.num_nodes**2    
       
       if self.do_train:
          with tf.name_scope('input_train'):
               self.encoded_protein = tf.compat.v1.placeholder(dtype=tf.float32, shape=(self.num_nodes, self.len_sequence))
               self.bias_in1 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train1')
               self.bias_in2 = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train2')
               self.lbl_in1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.lbl_in2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.msk_in1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.msk_in2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.neg_msk1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.neg_msk2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))         
       else:    
          with tf.name_scope('input_train'):
               self.encoded_protein = tf.compat.v1.placeholder(dtype=tf.float32, shape=(self.num_nodes, self.len_sequence))
               self.bias_in = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train')
               self.lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
               self.neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))            
       
       self.instantiate_embeddings()
       self.logits = self.inference()
       self.loss, self.accuracy = self.loss_func()
       self.train_op = self.train()
       
    def inference(self):
        embedding_proteins = self.word2sentence()  
        embedding_proteins = tf.nn.l2_normalize(embedding_proteins, 1)   
           
        self.model = GraphConvolutionSparse(
                        n_node = embedding_proteins.shape[0].value,
                        input_dim = embedding_proteins.shape[1].value,
                        output_dim = 128,
                        act=tf.nn.leaky_relu,
                        dropout = 0.25)
        if self.do_train:
           self.final_embedding1 = self.model.encoder(embedding_proteins, self.bias_in1)
           self.final_embedding2 = self.model.encoder(embedding_proteins, self.bias_in2)
           self.logits1 = self.model.decoder(self.final_embedding1, self.num_nodes)
           self.logits2 = self.model.decoder(self.final_embedding2, self.num_nodes)
           logits = (self.logits1 + self.logits2)/2
           self.final_embedding = (self.final_embedding1 + self.final_embedding2)/2
           return logits
        else:   
           self.final_embedding = self.model.encoder(embedding_proteins, self.bias_in)
           logits = self.model.decoder(self.final_embedding, self.num_nodes)
           return logits
    
    def loss_func(self):
        if self.do_train:
           loss1 = masked_accuracy(self.logits1, self.lbl_in1, self.msk_in1, self.neg_msk1)
           loss2 = masked_accuracy(self.logits2, self.lbl_in2, self.msk_in2, self.neg_msk2)
           loss = (loss1 + loss2)/2
           accuracy = loss
        else:    
           loss = masked_accuracy(self.logits, self.lbl_in, self.msk_in, self.neg_msk)
           accuracy  =  loss
        return loss, accuracy
    
    def train(self):
        train_op = self.model.training(self.loss, self.lr, self.l2_coef)
        return train_op
    
    ############  --- convolution ---  #################
    def word2sentence(self):
        return self.conv1dim(tf.cast(self.encoded_protein,dtype=tf.int32))
    
    def conv1dim(self, batch_protein):
        embedding_protein = tf.nn.embedding_lookup(self.embedding_tokens, batch_protein) 
        embedding_protein = tf.layers.conv1d(embedding_protein, 16, 10, use_bias=True, padding = "valid", activation='relu') 
        embedding_protein = tf.layers.max_pooling1d(embedding_protein, pool_size=60, strides=60)  
        final_embedding_protein = tf.contrib.layers.flatten(embedding_protein)  
        return final_embedding_protein
    
    def instantiate_embeddings(self):
        """define all embeddings here"""
        with tf.name_scope("token_embedding"):  
           self.embedding_tokens = tf.get_variable("embedding", shape=[self.token_size, self.dim_embedding],initializer=tf.random_normal_initializer(stddev=0.1))     
            
    
