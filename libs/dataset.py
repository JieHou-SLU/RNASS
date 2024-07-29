from spektral.data.loaders import BatchLoader
from spektral.data import Dataset, Graph
import numpy as np
import scipy.sparse as sp
import glob
import sys
import re

from tensorflow.keras.utils import *
import tensorflow as tf
epsilon = tf.keras.backend.epsilon()
from tensorflow.keras import backend as K
from libs.utils import *

class gcn_data_wrapper(Dataset):
    def __init__(self, name, dataset, expected_n_channels, edge_type, **kwargs):
        self.name = name.lower()
        self.expected_n_channels = expected_n_channels
        self.edge_type = edge_type
        
        # sort table by length
        #self.dataset = dataset.sort_values(["Length"], ascending=True).reset_index(drop=True)
        self.dataset = dataset.sample(frac=1).reset_index(drop=True)
        
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        #for index in tqdm(self.dataset.index, desc='Loading data'):
        for index in self.dataset.index:
            
            batch_data = self.dataset.iloc[index: (index + 1)] # select rows of dataframe

            X, EE_val, EE_adj, Y, nt_Y = get_input_output_rna_augment_bin_ntApairRegularized_gcn(batch_data, self.expected_n_channels, edge_type= self.edge_type)

            # single sample
            x = X[0].astype(np.float32)
            a = EE_adj[0].astype(np.float32)
            e = EE_val[0].astype(np.float32)
            y = Y[0]
            y_nt = nt_Y[0]
            
            graph = Graph(x=x, a=a, e = e, y=[y, y_nt, a])
            graphs.append(graph)
        return graphs


class RnaGenerator_gcn_sp(Sequence):
    def __init__(self, dataset, batch_size, expected_n_channels, edge_type, transforms=None):
        self.batch_size = batch_size
        self.expected_n_channels = expected_n_channels
        self.edge_type = edge_type
        self.transforms = transforms
        
        # sort table by length
        self.dataset = dataset.sort_values(["Length"], ascending=True).reset_index(drop=True)

    def on_epoch_begin(self):
        self.indexes = np.arange(len(self.dataset))
        #np.random.shuffle(self.indexes)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        batch_data = self.dataset.iloc[index * self.batch_size: (index + 1) * self.batch_size] # select rows of dataframe

        dataset = gcn_data_wrapper(name = 'tmp', dataset = batch_data, expected_n_channels = self.expected_n_channels, edge_type = self.edge_type, transforms=self.transforms)

        return dataset


def get_feature_and_y_ntApairRegularized_gcn(batch_data, i, expected_n_channels, include_pseudoknots=True, edge_type = 'LinearPartition'):
        sequence = batch_data['Sequence'][i]
    
        if include_pseudoknots:
            pairing_list = batch_data['BasePairs'][i].split(',')
        else:
            pairing_list = batch_data['UnknottedPairs'][i].split(',')
        
        linearProb = batch_data['LinearPartition'][i]
        RNAfold_pairs = batch_data['RNAfold_ss'][i]
        Contrafold_pairs = batch_data['Contrafold_ss'][i]
        Centroidfold_pairs = batch_data['Centroidfold_ss'][i]
        MXfold_pairs = batch_data['MXfold_ss'][i]

        ###########################################   Extract pairs
        nt_types = ['A','U','C','G']
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
        
        one_hot_feat = one_hot(sequence)
        seqLen = len(sequence)
        
        encode_feature = one_hot_feat
        
        label_mask = l_mask(one_hot_feat, seqLen)
    
        true_contact = np.zeros((seqLen, seqLen))
        nt_contact = np.zeros((seqLen, seqLen))
        
        for i in range(0,seqLen):
            for j in range(0,seqLen):
                xx = 0
                if i == j:
                    xx = 0
                if str(i+1)+"-"+str(j+1) in pairing_list or str(j+1)+"-"+str(i+1) in pairing_list:
                    xx = 1 
                true_contact[i, j] = xx
                true_contact[j, i] = xx
                
                tt = 0
                if i == j:
                    tt = 0
                if sequence[i]+sequence[j] in include_pairs:
                    tt = 1 
                nt_contact[i, j] = tt
                nt_contact[j, i] = tt
    
        true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
        # i - j >= 2
        true_contact = np.tril(true_contact, k=-2) + np.triu(true_contact, k=2) # remove the diagnol contact
        true_contact = true_contact.astype(np.uint8)
    
        nt_contact[nt_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
        # i - j >= 2
        nt_contact = np.tril(nt_contact, k=-2) + np.triu(nt_contact, k=2) # remove the diagnol contact
        nt_contact = nt_contact.astype(np.uint8)


        node_feature = encode_feature

        if edge_type == 'LinearPartition':
            ss_energy = linearProb
        else: 
            ss_energy = MXfold_pairs
        if ss_energy == '':
            ss_energy = linearProb 
        if ss_energy[0] == ',':
            ss_energy = ss_energy[1:]
        if ss_energy[0] == ';':
            ss_energy = ss_energy[1:]

        lines = ss_energy.split(',')
        
        nt_pair_prob = np.zeros((seqLen, seqLen))
        
        for line in lines:
            s = line.strip().replace(' ','') # 1-22-0.5
            # Use regex to correctly match all numbers, including scientific notation
            # The pattern matches numbers possibly followed by scientific notation
            arr = re.findall(r'\d+\.?\d*(?:e-?\d+)?', s)
            if not line.startswith('#') and len(arr)>=2:
                #print(arr)
                nt_index = int(arr[0])
                pair_index = int(arr[1])
                if len(arr)>2:
                    pair_prob = float(arr[2]) # ss prob
                else:
                    pair_prob = 1 # ss pairs
                nt_pair_prob[nt_index-1,pair_index-1] = pair_prob
                nt_pair_prob[pair_index-1,nt_index-1] = pair_prob

        
        edge_feature = np.copy(nt_pair_prob)
        edge_feature[edge_feature>0.5]=1
        
        ### add physicochemical_property_indices (L,L,22)
        physicochemical = np.zeros((seqLen, seqLen, 22))
        for i in range(0,seqLen):
            for j in range(0,seqLen):
                nt_pair = sequence[i]+sequence[j]
                if nt_pair in physicochemical_property_indices:
                    physicochemical[i, j] = physicochemical_property_indices[nt_pair] 
                else:
                    physicochemical[i, j] = 0
                    #raise Exception(nt_pair,' is not found in physicochemical_property_indices')

        edge_val_feature = physicochemical
        
        # should we set diagnal to 1?
        np.fill_diagonal(edge_feature, 1)        
   
        # should we set offset diagnal to 1?
        rng = np.arange(len(edge_feature)-1)
        edge_feature[rng, rng+1] = 1
        edge_feature[rng+1, rng] = 1
        

        ###########################################   Extract pairs    

        # Create X and Y placeholders
        # Sequence features
        X = np.full((seqLen, expected_n_channels), 0)
        X[:, 0:expected_n_channels] = node_feature


        E_val = np.full((seqLen, seqLen, 22), 0)
        E_val[:, :, 0:22] = edge_val_feature
        
        E_adj = np.full((seqLen, seqLen), 0)
        E_adj[:, :] = edge_feature
    
    
        Y0 = np.full((seqLen, seqLen), 0)
        nt_Y0 = np.full((seqLen, seqLen), 0)
    
        # label
        Y0[:, :] = true_contact
        
        # nt label
        nt_Y0[:, :] = nt_contact
        return X, E_val, E_adj, Y0, nt_Y0


def get_input_output_rna_augment_bin_ntApairRegularized_gcn(batch_data, expected_n_channels, edge_type = 'LinearPartition'):
    # get maximum length
    OUTL = batch_data["Length"].max()

    #### find the dimension
    total_dim = len(batch_data)
    
    #### Define node matrix
    XX = np.full((total_dim, OUTL, expected_n_channels), 0)
    
    #### Define edge feature matrix
    EE_val = np.full((total_dim, OUTL, OUTL, 22), 0)
    
    
    #### Define edge adjacency matrix
    EE_adj = np.full((total_dim, OUTL, OUTL), 0)
    
    
    #### Define output
    YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    nt_YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    pair_YY = np.full((total_dim, OUTL, OUTL, 1), 0)
    
    
    indx = 0
    
    for i in batch_data.index:
        rna = batch_data['RNA_ID'][i]
        
        X, E_val, E_adj,Y0,nt_Y0 = get_feature_and_y_ntApairRegularized_gcn(batch_data, i, expected_n_channels, edge_type == edge_type)

        assert len(X[0, :]) == expected_n_channels
        assert len(X[:, 0]) >= len(Y0[:, 0])
        if len(X[:, 0]) != len(Y0[:, 0]):
            print('')
            print('WARNING!! Different len(X) and len(Y) for ', pdb, len(X[:, 0, 0]), len(Y0[:, 0]))
        
        l = len(X[:, 0])
        XX[indx, :l, :] = X 
        EE_val[indx, :l, :l, :] = E_val
        EE_adj[indx, :l, :l] = E_adj
        
        YY[indx, :l, :l, 0] = Y0
        nt_YY[indx, :l, :l, 0] = nt_Y0
        indx += 1
  
    return XX.astype(np.float32), EE_val.astype(np.float32), EE_adj.astype(np.float32), YY.astype(np.float32), nt_YY.astype(np.float32)

