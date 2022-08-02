import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
import tensorflow as tf
import networkx as nx
from rdkit import Chem

def load_data_DDI():
    labels = np.loadtxt("data/adj_ddi_pretrain.txt")
    smiles_file = "data/id_smiles.txt" 
    molecular_graph_list = process(smiles_file)
    encoded_drugs = transform_graph_to_sequence(molecular_graph_list)
    mol_graphs = construct_molecular_graph(molecular_graph_list)
    
    interaction = sio.loadmat('data/interaction_ddi.mat') 
    interaction = interaction['interaction']
    
    logits_train = interaction
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    
    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)
    return interaction, logits_train, train_mask, labels, encoded_drugs, mol_graphs

def load_data_for_fine_tuning(train_arr, test_arr):
    labels = np.loadtxt("data/adj_ddi_fine_tuning.txt")
    
    num_node = 1971
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(num_node, num_node)).toarray()
    logits_test = logits_test.reshape([-1,1])  

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(num_node, num_node)).toarray()
    logits_train = logits_train + logits_train.T
    interaction = logits_train
    logits_train = logits_train.reshape([-1,1])
      
    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])
    
    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)
    return interaction, logits_train, logits_test, train_mask, test_mask, labels

def load_drug_graph():
    smiles_file = "data/id_smiles.txt" 
    molecular_graph_list = process(smiles_file)
    encoded_drugs = transform_graph_to_sequence(molecular_graph_list)
    mol_graphs = construct_molecular_graph(molecular_graph_list)
    return encoded_drugs, mol_graphs

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def construct_molecular_graph(graph_list):
    graphs = dict()
    temp = np.zeros([10,10])
    for key in graph_list.keys():
        graph_mol = graph_list[key]
        if graph_mol==0:
           graphs[key] = temp
           continue
        symbols = nx.get_node_attributes(graph_mol, 'symbol')
        num_node = len(list(symbols.keys()))
        matrix = np.zeros([num_node, num_node])
        bonds = nx.get_edge_attributes(graph_mol, 'bond_type')
        edge_index = list(bonds.keys())
        for edge in edge_index:
            x, y = int(edge[0]-1), int(edge[1]-1)
            matrix[x,y]=1
            matrix[y,x]=1
        graphs[key] = matrix
    return graphs     
        
def process(file_name):
    id_graph_dict = dict()
    with open(file_name) as f:
        for line in f:
            line = line.rstrip()
            id, smiles = line.split()
            if smiles=='0':
                id_graph_dict[id] = 0
                continue
            mol = Chem.MolFromSmiles(smiles)               
            if mol==None:
                id_graph_dict[id] = 0
                continue
            graph = mol_to_nx(mol)
            id_graph_dict[id] = graph
    return id_graph_dict  

def symbol_mapping():
    # generate the node feature for each symbol(element)
    # we have 22 different elements in this data set we use the one hot vector
    # or fix_dim 8 dim vector to represent each symbol.
    #num_symbols = 40
    symbol_dict = dict()
    keys = ['Ag','Al','As','Au','B','Bi','Br','C','Ca','Cl','Co','Cr','Cu','F','Fe','Ga','Gd','H','Hg','I','K','La','Li','Lu',
            'Mg','Mn','Mo','N','Na','O','P','Pt','Ra','S','Se','Si','Sn','Sr','Tc','Ti','Xe','Zn','Zr']
    for i in range(len(keys)):
        symbol_dict[keys[i]] = i
    return symbol_dict

def transform_graph_to_sequence(graph_list):
    sequences = []
    for key in graph_list.keys():
        seq = []
        graph = graph_list[key]
        if graph==0:
            sequences.append([0]*10)
            continue
        symbols = nx.get_node_attributes(graph, 'symbol')
        symbol_dict = symbol_mapping()
        for sym in symbols.keys():
            s = symbols[sym]
            seq.append(symbol_dict[s])
        sequences.append(seq)    
    return sequences     
        
        
    