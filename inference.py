import os
import sys
import numpy as np
import datetime
import argparse
import random
import operator
import itertools
import operator
import pandas as pd

from libs.utils import *
from libs.model import * 
from libs.evaluation import *
from libs.dataset import * 

flag_plots = False

if flag_plots:
    #%matplotlib inline
    from plots import *

if sys.version_info < (3,0,0):
    print('Python 3 required!!!')
    sys.exit(1)


import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# use cpu only
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm
from spektral.data.loaders import BatchLoader
from spektral.data import Dataset, Graph
import numpy as np
import scipy.sparse as sp
import glob



def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter,
            epilog='EXAMPLE:\npython3 inference.py -m models/LinearPartition_use/ARMAConv/rna_best_val.hdf5 -p data/Refined_dataset.h5 -o results/test_ARMA -w ARMAConv -a LinearPartition')
    parser.add_argument('-m', type=str, required = True, dest = 'file_weights', help="hdf5 weights file")
    parser.add_argument('-b', type=int, required = True, dest = 'batch_size', help="number of pdbs to use for each batch")
    parser.add_argument('-r', type=str, required = True, dest = 'len_range', help="lengths of pdbs to use for each batch")
    parser.add_argument('-e', type=int, required = True, dest = 'training_epochs', help="# of epochs")
    parser.add_argument('-o', type=str, required = True, dest = 'dir_out', help="directory to write .npy files")
    parser.add_argument('-d', type=int, required = True, dest = 'arch_depth', help="residual arch depth")
    parser.add_argument('-f', type=int, required = True, dest = 'filters_per_layer', help="number of convolutional filters in each layer")
    parser.add_argument('-p', type=str, required = True, dest = 'data_path', help="path where all the data (including .lst) is located")
    parser.add_argument('-k', type=int, required = True, dest = 'filter_size_2d', help="filter_size_2d")
    parser.add_argument('-l', type=float, required = True, dest = 'loss_ratio', help="ratio in weighted loss")
    parser.add_argument('-t', type=float, required = True, dest = 'dropout_rate', help="ratio in dropout")
    parser.add_argument('-z', type=int, required = True, dest = 'lstm_layers', help="number of lstm layers")
    parser.add_argument('-y', type=int, required = True, dest = 'fully_layers', help="number of fully connected layers")
    parser.add_argument('-g', type=float, required = True, dest = 'nt_reg_weight', help="weight for nt regularized term")
    parser.add_argument('-u', type=float, required = True, dest = 'pair_reg_weight', help="weight for pair_reg_weight  regularized term")
    parser.add_argument('-j', type=int, required = True, dest = 'lstm_filter', help="filter size for lstm")
    parser.add_argument('-i', type=int, required = True, dest = 'include_pseudoknots', help="filter type")
    parser.add_argument('-q', type=int, required = True, dest = 'dilation_size', help="dilation")
    parser.add_argument('-w', type=str, required = True, dest = 'gcn_type', help="gcn_type")
    parser.add_argument('-a', type=str, required = True, dest = 'edge_type', help="edge_type")
    
    args = parser.parse_args()
    return args


import sys
sys.argv = ['script_name.py',  '-b', '8', '-r', '0-500', '-e', '50', '-d', '10', '-f', '40',
 '-k', '7', '-l', '0.5', '-t', '0.3', '-z', '2', '-y', '0', '-g', '1', '-u', '1', 
'-j', '8', '-q', '4', '-i', '1'] + sys.argv[1:]


#['GraphSage', 'DefaultGatedGCN', 'AGNNConv', 'APPNPConv', 'ARMAConv', 'EdgeConv', 'GATConv', 'GATConv4', 'GCNConv']


############################# (1) Define argument ############################################

args = get_args()

batch_size                = args.batch_size # 8
arch_depth                = args.arch_depth # 10
filters_per_layer         = args.filters_per_layer #40
len_range                 = args.len_range  # 0-500
training_epochs           = args.training_epochs #50  
data_path               = args.data_path #'~/data/Own_data/Sharear' 
dir_out                   = args.dir_out #' /train_results' 
filter_size_2d            = args.filter_size_2d # 7
length_start              = 0
length_end                = 500
if len(len_range.split('-')) == 2:
    length_start              = int(len_range.split('-')[0])
    length_end                = int(len_range.split('-')[1])

file_weights              = args.file_weights # rna_best_val.hdf5'

loss_ratio                = args.loss_ratio # 1

dropout_rate              = args.dropout_rate # 1
lstm_layers               = args.lstm_layers # 1
fully_layers              = args.fully_layers # 1
nt_reg_weight               = args.nt_reg_weight # 1
pair_reg_weight               = args.pair_reg_weight # 1
lstm_filter     = args.lstm_filter # 1
dilation_size =  args.dilation_size # 1
regularize            = True

edge_type = args.edge_type 

if edge_type not in ['LinearPartition', 'MXfold']:
    edge_type =  'MXfold'

'''
if args.include_pseudoknots == 1:
    include_pseudoknots = True
    print("Including pseudoknots")
else:
    include_pseudoknots = False
    print("Excluding pseudoknots")
'''
include_pseudoknots = True


expected_n_channels       = 4

gcn_type = args.gcn_type # 'GraphSage'

print("## Setting gcn_type to ",gcn_type)

print('Start ' + str(datetime.datetime.now()))

print('')
print('Parameters:')
print('file_weights', file_weights)
print('arch_depth', arch_depth)
print('filters_per_layer', filters_per_layer)
print('batch_size', batch_size)
print('data_path', data_path)
print('dir_out', dir_out)

os.system('mkdir -p ' + dir_out)



#############################  2. Define the model #################################

# Import after argparse because this can throw warnings with "-h" option
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Allow GPU memory growth
if hasattr(tf, 'GPUOptions'):
    from tensorflow.python.keras import backend as K
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    #K.tensorflow_backend.set_session(sess)
    K.set_session(sess)
else:
    # For other GPUs
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

print('')
print('Build a model..')
model = ''
#gcn_type = 'DefaultGatedGCN'
model = rna_pair_prediction_bin_spektral(node_dim=expected_n_channels, gcn_type = gcn_type, num_gcn_layers = arch_depth, filter_size = filter_size_2d, num_lstm_layers = lstm_layers, hidden_dim = filters_per_layer, regularize=regularize, dropout_rate = dropout_rate, dilation_size=1)

print('')
print('Compile model..')



losses = {
    "nt_out": weighted_binary_crossentropy_ntRegularized,
  "pair_out2": weighted_binary_crossentropy_pairRegularized
}


optimizer = tf.keras.optimizers.Adam(0.001, decay=2.5e-4)

lossWeights = {"pair_out2": pair_reg_weight, "nt_out": nt_reg_weight}

model.compile(
    optimizer=optimizer, # you can use any other optimizer
    loss = losses, #loss='binary_crossentropy',
    loss_weights=lossWeights,
    metrics=[
        precision,
        recall,
        f1,
        specificity,
        mcc,
        tp,
        pp,
        data_size        
    ]
)

print(model.summary(line_length=150))


######################################################################### 3. Define the dataset 

seq_len_range = (length_start,length_end)
dataset = pd.read_hdf(data_path, "df").query("Length >= {} and Length <= {}".format(*seq_len_range)).reset_index(drop=True)

dataset['StructureEnergy'] = dataset['StructureEnergy'].astype(float)


### Load bpRNA data
bpRNA_train_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Train') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)
bpRNA_valid_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Validation') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)
bpRNA_test_data = dataset[((dataset['DataSource'] == 'bpRNA') & (dataset['DataType']== 'Test') & (dataset["StructureEnergy"]<-10))].reset_index(drop=True)

### Load PDB data
pdb_train_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Train') )].reset_index(drop=True)
pdb_valid_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Validation'))].reset_index(drop=True)
pdb_test_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test') )].reset_index(drop=True)
pdb_test2_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test2'))].reset_index(drop=True)
pdb_test3_data = dataset[((dataset['DataSource'] == 'PDB') & (dataset['DataType']== 'Test_hard'))].reset_index(drop=True)



data_transform = None
if gcn_type == 'GraphSage':    
    data_transform = [LayerPreprocess(GraphSageConv), AdjToSpTensor()]
elif gcn_type == 'AGNNConv':
    data_transform = [LayerPreprocess(AGNNConv), AdjToSpTensor()]
elif gcn_type == 'APPNPConv':
    data_transform = [LayerPreprocess(APPNPConv), AdjToSpTensor()]
elif gcn_type == 'ARMAConv':
    data_transform = [LayerPreprocess(ARMAConv), AdjToSpTensor()]
elif gcn_type == 'EdgeConv':
    data_transform = [LayerPreprocess(EdgeConv), AdjToSpTensor()]
elif gcn_type == 'GATConv':
    data_transform = [LayerPreprocess(GATConv), AdjToSpTensor()]
elif gcn_type == 'GATConv4':
    data_transform = [LayerPreprocess(GATConv), AdjToSpTensor()]
elif gcn_type == 'GatedGraphConv':
    data_transform = [LayerPreprocess(GatedGraphConv), AdjToSpTensor()]
elif gcn_type == 'GCNConv':
    data_transform = [LayerPreprocess(GCNConv), AdjToSpTensor()]
else:
    data_transform = [LayerPreprocess(GraphSageConv), AdjToSpTensor()]

bpRNA_train_generator = RnaGenerator_gcn_sp(bpRNA_train_data, batch_size, expected_n_channels, edge_type = edge_type, transforms=data_transform)
bpRNA_valid_generator = RnaGenerator_gcn_sp(bpRNA_valid_data, batch_size, expected_n_channels, edge_type = edge_type, transforms=data_transform)

PDB_train_generator = RnaGenerator_gcn_sp(pdb_train_data, batch_size, expected_n_channels, edge_type = edge_type,transforms=data_transform)
PDB_valid_generator = RnaGenerator_gcn_sp(pdb_valid_data, batch_size, expected_n_channels, edge_type = edge_type,transforms=data_transform)


print('')
print('len(bpRNA_train_generator) : ' + str(len(bpRNA_train_generator)))
print('len(bpRNA_valid_generator) : ' + str(len(bpRNA_valid_generator)))


graph = bpRNA_train_generator[1]
X = graph[0].x
EE_adj = graph[0].a
EE_val = graph[0].e
Y = graph[0].y[0]
Y_nt = graph[0].y[1]

print('Actual shape of X    : ' + str(X.shape))
print('Actual shape of EE_val    : ' + str(EE_val.shape))
print('Actual shape of EE_adj    : ' + str(EE_adj.shape))
print('Actual shape of Y    : ' + str(Y.shape))
print('Actual shape of Y_nt    : ' + str(Y_nt.shape))

print("Y==1: ", np.count_nonzero(Y == 1))
print("Y==0: ", np.count_nonzero(Y == 0))
print("Y==-1: ", np.count_nonzero(Y == -1))


 
eva_dataset = pd.concat([bpRNA_valid_data, bpRNA_test_data,pdb_train_data,pdb_valid_data,pdb_test_data,pdb_test2_data,pdb_test3_data], ignore_index=True).reset_index(drop=True)


print('')
#print('Channel summaries:')
#summarize_channels(X[0, :, :], Y[0])


if os.path.exists(file_weights):
    print('')
    print('Loading existing weights..')
    try:
        model.load_weights(file_weights)
    except:
        print("Loading model error!")



# Setup the ModelCheckpoint callback

checkpoint_callback = ModelCheckpoint(
    filepath=file_weights,
    monitor='pdb_val_loss',  # Change to your validation F1 or other metric as necessary
    save_best_only=True,
    save_weights_only=True,
    mode='min',  # Change based on the nature of monitor ('min' for loss, 'max' for accuracy)
    verbose=1
)

# Initialize the callback
checkpoint_callback.set_model(model)
#os.makedirs(dir_out + '/pred_ct_files/', exist_ok=True)


# evaluate

bpRNA_acc_record_val=[]
bpRNA_acc_record_test=[]
pdb_acc_record_train=[]     
pdb_acc_record_val=[]    
pdb_acc_record_test=[]        
pdb_acc_record_test2=[]     
pdb_acc_record_test3=[]      

print("Evaluating on all data with ", len(eva_dataset), " rnas\n")

model_pred = rna_pair_prediction_bin_spektral(node_dim=expected_n_channels, gcn_type = gcn_type, num_gcn_layers = arch_depth, filter_size = filter_size_2d, num_lstm_layers = lstm_layers, hidden_dim = filters_per_layer, regularize=regularize, dropout_rate = dropout_rate, dilation_size=1)

print("Loading model: ",file_weights)
model_pred.load_weights(file_weights)  

idx_num = 0
results_summary = pd.DataFrame(columns = ['pdbid', 'source', 'f1-score'])
with tqdm(total=len(eva_dataset), desc=f"Evaluation Dataset") as pbar:
    for rna_id in eva_dataset.index:
        #print(str(rna_id)+",", end="", flush=True)

        rna = eva_dataset['RNA_ID'][rna_id]
        rna_datasource = eva_dataset['DataSource'][rna_id]
        rna_datatype = eva_dataset['DataType'][rna_id]
        sequence = eva_dataset['Sequence'][rna_id]
        if include_pseudoknots:
            pairing_list = eva_dataset['BasePairs'][rna_id].split(',')
        else:
            pairing_list = eva_dataset['UnknottedPairs'][rna_id].split(',')
        
        one_hot_feat = one_hot(sequence)
        label_mask = l_mask(one_hot_feat, len(sequence))
        
        batch_data = eva_dataset.iloc[rna_id: (rna_id + 1)] # select rows of dataframe
        tmp_dataset = gcn_data_wrapper(name = 'tmp', dataset = batch_data, expected_n_channels = expected_n_channels, edge_type = edge_type, transforms=data_transform)
        graph = tmp_dataset[0]
        #x, a, y = graph.x, graph.a, graph.y     
        X = tf.expand_dims(graph.x, axis=0) 
        EE_val = tf.expand_dims(graph.e, axis=0) 
        EE_adj = tf.expand_dims(graph.y[2], axis=0) 
        EE_adj2 = graph.a
        Y = tf.expand_dims(graph.y[0], axis=0) 
        Y_nt = tf.expand_dims(graph.y[1], axis=0) 

        # evaluate

        output_prob, _ = model_pred([X,EE_val,EE_adj,EE_adj2], training=False)
        output_prob = output_prob[0] # get first sample output
        
        idx_num += 1
        output_class = output_prob > 0.5
        seqLen = len(sequence)
        true_contact = np.zeros((seqLen, seqLen))
        for i in range(0,seqLen):
            for j in range(0,seqLen):
                xx = 0
                if i == j:
                    xx = 0
                if str(i+1)+"-"+str(j+1) in pairing_list or str(j+1)+"-"+str(i+1) in pairing_list:
                    xx = 1 
                true_contact[i, j] = xx
                true_contact[j, i] = xx
    
        true_contact[true_contact < 0] = 0 # transfer -1 to 0, shape = (L, L)
        # i - j >= 2
        true_contact = np.tril(true_contact, k=-2) + np.triu(true_contact, k=2) # remove the diagnol contact
        true_contact = true_contact.astype(np.uint8)

        acc = evaluate_predictions_single(true_contact, output_prob[:,:,0],sequence,label_mask)
        
        predicted_structure = get_ss_pairs_from_matrix(output_prob[:,:,0],sequence,label_mask, 0.5)
        
        #ct_out = dir_out + '/pred_ct_files/' + rna + '.ct'
        #ct_file_output(predicted_structure,sequence,ct_out)
        
        new_rows = pd.DataFrame([
                {'pdbid': rna, 'source': rna_datasource, 'f1-score': str(acc[4]), }
            ])
        results_summary = pd.concat([results_summary, new_rows], ignore_index=True)
        if rna_datasource == 'bpRNA' and rna_datatype == 'Validation':
              bpRNA_acc_record_val.append(acc) 
        if rna_datasource == 'bpRNA' and rna_datatype == 'Test':
              bpRNA_acc_record_test.append(acc) 
        if rna_datasource == 'PDB' and rna_datatype == 'Train':
              pdb_acc_record_train.append(acc) 
        if rna_datasource == 'PDB' and rna_datatype == 'Validation':
              pdb_acc_record_val.append(acc) 
        if rna_datasource == 'PDB' and rna_datatype == 'Test':
              pdb_acc_record_test.append(acc) 
        if rna_datasource == 'PDB' and rna_datatype == 'Test2':
              pdb_acc_record_test2.append(acc) 
        if rna_datasource == 'PDB' and rna_datatype == 'Test_hard':
              pdb_acc_record_test3.append(acc) 

        # Update tqdm progress bar message
        pred_f1 = acc[4]
        pbar.set_postfix(rna_idx = rna_id, f1_score=pred_f1)
        pbar.update(1)  # Manually increment the progress bar
        
results_summary.to_csv(dir_out+'/data_evaluation_summary.txt', index=False)

print("\n\n########################### bpRNA validation evaluation (total ",len(bpRNA_valid_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(bpRNA_acc_record_val), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### bpRNA test evaluation (total ",len(bpRNA_test_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(bpRNA_acc_record_test), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### PDB train evaluation (total ",len(pdb_train_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(pdb_acc_record_train), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### PDB validation evaluation (total ",len(pdb_valid_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(pdb_acc_record_val), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### PDB test evaluation (total ",len(pdb_test_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(pdb_acc_record_test), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### PDB test2 evaluation (total ",len(pdb_test2_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(pdb_acc_record_test2), axis=0).round(decimals=5)
output_result_simple(avg_acc)


print("\n\n########################### PDB test_hard evaluation (total ",len(pdb_test3_data)," rnas): ###################################\n\n")
avg_acc = np.mean(np.array(pdb_acc_record_test3), axis=0).round(decimals=5)
output_result_simple(avg_acc)
    
print("\n\n#######################################################################\n\n")


