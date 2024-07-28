from tensorflow.keras.utils import *
import tensorflow as tf

from tensorflow.keras import backend as K
epsilon = tf.keras.backend.epsilon()
import matplotlib
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.python.keras import layers, optimizers

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Input, Conv1D, Convolution2D, Activation, add, Dropout, BatchNormalization, Reshape, Lambda,Bidirectional
from tensorflow.keras.layers import Flatten, Dense, Embedding, concatenate, Add, Multiply, Concatenate, ConvLSTM2D

from spektral.transforms import AdjToSpTensor, LayerPreprocess
# Create the model using a Spektral layer
from spektral.layers import GraphSageConv, AGNNConv, APPNPConv, ARMAConv,EdgeConv,GATConv,GatedGraphConv, GCNConv

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2



def ReshapeConv_to_LSTM(x):
    reshape=K.expand_dims(x,0)
    return reshape

def ReshapeLSTM_to_Conv(x):
    reshape=K.squeeze(x,0)
    return reshape



def rna_pair_prediction_bin_spektral(gcn_type = 'ARMAConv', node_num = None, node_dim=4, hidden_dim=100, voc_edges_in = 2, voc_edges_out = 1, voc_nodes_out = 2, num_gcn_layers = 10, num_lstm_layers = 2, lstm_filter = 8, aggregation = "mean", regularize=False, dropout_rate = 0.25, dilation_size = 1, filter_size=3):
    if  gcn_type not in ['DefaultGatedGCN', 'APPNPConv', 'ARMAConv',  'EdgeConv', 'GATConv', 'GatedGraphConv', 'GCNConv']:
        print("Wrong gcn_type")
        raise

    print("### Setting gcn_type to ",gcn_type)
    dropout_value = dropout_rate
    
    # (1) Node embedding
    node_input = Input(shape = (node_num,node_dim))
    nodes_embedding = InstanceNormalization()(node_input) # B x V x H
    nodes_embedding = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = dilation_size, kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(nodes_embedding)
    nodes_embedding = Activation('relu')(nodes_embedding)
    nodes_embedding = BatchNormalization()(nodes_embedding)
    nodes_embedding = Dropout(dropout_value)(nodes_embedding)
    
    
    # (2) Edge weight embedding: Input edge distance matrix (batch_size, num_nodes, num_nodes)

    edges_value_input = Input(shape = (node_num,node_num,22))
    edges_value_embedding = InstanceNormalization()(edges_value_input)

    edges_value_embedding = Convolution2D(hidden_dim//2, kernel_size = (filter_size, filter_size), dilation_rate = (dilation_size,dilation_size), kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(edges_value_embedding)
    edges_value_embedding = Activation('relu')(edges_value_embedding)
    edges_value_embedding = BatchNormalization()(edges_value_embedding)
    edges_value_embedding = Dropout(dropout_value)(edges_value_embedding)
  
    
    # (3) Input edge adjacency matrix (batch_size, num_nodes, num_nodes)
    edges_adj_input = Input(shape = (node_num,node_num))

    # (4) Convert sparse input to dense
    edges_adj_embedding = Embedding(voc_edges_in, hidden_dim//2)(edges_adj_input) # B x V x V x H


    # (5) merge edge embedding
    edges_value_embedding = concatenate([edges_value_embedding, edges_adj_embedding])

    
    # (6) Copy edge adjacency matrix (batch_size, num_nodes, num_nodes) for node convolution
    edges_adj_input2 = Input(shape=(None,), dtype=tf.float32, sparse=True, name='A_in') # follow spektral's requirement
    
    ################ (1) Define Gated GNN Layer ################
    d_rate = dilation_size
    for layer in range(num_gcn_layers):
        x_in,e_in,a_in = nodes_embedding, edges_value_embedding, edges_adj_input2

          
        ################# (1.1) Edge convolution (class EdgeFeatures(nn.Module))
        """Convnet features for edges.
        e_ij = U*e_ij + V*(x_i + x_j)
        """
        edge_Ue = Convolution2D(hidden_dim, kernel_size = (filter_size, filter_size), dilation_rate = (d_rate,d_rate), kernel_initializer = 'he_normal', kernel_regularizer = l2(0.0001), padding = 'same')(e_in)
        edge_Ue = Activation('relu')(edge_Ue)
        edge_Ue = BatchNormalization()(edge_Ue)
        edge_Ue = Dropout(dropout_value)(edge_Ue)
        
        
        edge_Vx = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        edge_Vx = Activation('relu')(edge_Vx)
        edge_Vx = BatchNormalization()(edge_Vx)
        edge_Vx = Dropout(dropout_value)(edge_Vx)
        
        edge_Wx = K.expand_dims(edge_Vx, 2) # Extend Vx from "B x V x H" to "B x V x 1 x H"
        edge_Vx = K.expand_dims(edge_Vx, 1) # extend Vx from "B x V x H" to "B x 1 x V x H"
        
        '''
        e_new: Convolved edge features (batch_size, num_nodes, num_nodes, hidden_dim)
        '''
        #e_new = Ue + Vx + Wx
        edge_convnet = Add()([edge_Ue, edge_Vx, edge_Wx]) # B x V x V x H
        #####################################################################

        #'EdgeConv, APPNPConv, DefaultGatedGCN, ARMAConv, GATConv4,  GCNConv'

        node_combine = []
        #for gcn_type in ['GraphSage','APPNPConv', 'ARMAConv', 'EdgeConv',  'GATConv4',  'GatedGraphConv', 'GCNConv']:
        #for gcn_type in ['APPNPConv', 'ARMAConv', 'GCNConv', 'EdgeConv', 'GATConv4']:
        for gcn_type in [gcn_type]:
            if gcn_type == 'APPNPConv':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                node_convnet = APPNPConv(channels = hidden_dim, activation='relu')([node_in, a_in])
                
                node_convnet2 = K.expand_dims(node_convnet, 0)
    
            elif gcn_type == 'ARMAConv':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                node_convnet = ARMAConv(channels = hidden_dim, order=2, iterations=5, activation='relu')([node_in, a_in])
                node_convnet2 = K.expand_dims(node_convnet, 0)
    
            elif gcn_type == 'EdgeConv':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                #edge_in = tf.squeeze(edge_convnet, axis=0) 
                node_convnet = EdgeConv(channels = hidden_dim, aggregate='sum', activation='relu')([node_in, a_in])
                node_convnet2 = K.expand_dims(node_convnet, 0)
    
            elif gcn_type == 'GATConv':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                #edge_in = tf.squeeze(edge_convnet, axis=0) 
                node_convnet = GATConv(channels = hidden_dim, attn_heads=1, activation='relu')([node_in, a_in])
                node_convnet2 = K.expand_dims(node_convnet, 0)
            
            elif gcn_type == 'GATConv4':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                #edge_in = tf.squeeze(edge_convnet, axis=0) 
                node_convnet = GATConv(channels = hidden_dim, attn_heads=4, concat_heads=False, activation='relu')([node_in, a_in])
                node_convnet2 = K.expand_dims(node_convnet, 0)
    
            elif gcn_type == 'GCNConv':
                node_in = tf.squeeze(x_in, axis=0) 
                #adj_in = tf.squeeze(a_in, axis=0) 
                #edge_in = tf.squeeze(edge_convnet, axis=0) 
                node_convnet = GCNConv(channels = hidden_dim, activation='relu')([node_in, a_in])
                node_convnet2 = K.expand_dims(node_convnet, 0)

            node_combine.append(node_convnet2)
        #elif gcn_type == 'DefaultGatedGCN':

        ################# (1.2) Compute edge gates ######################### 
        edge_convnet_gate = Activation('sigmoid')(edge_convnet) # B x V x V x H
    
        ################# (1.3) Node convolution
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
            edge_gate: Edge gate values (batch_size, num_nodes, num_nodes, hidden_dim)
    
        Returns:
            node_convnet: Convolved node features (batch_size, num_nodes, hidden_dim)
        """
    
        """Convnet features for nodes.
    
        Using `sum` aggregation:
            x_i = U*x_i +  sum_j [ gate_ij * (V*x_j) ]
    
        Using `mean` aggregation:
            x_i = U*x_i + ( sum_j [ gate_ij * (V*x_j) ] / sum_j [ gate_ij] )
        """
        node_Ux = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        node_Ux = BatchNormalization()(node_Ux)
        node_Ux = Activation('relu')(node_Ux)
        node_Ux = Dropout(dropout_value)(node_Ux)
        
        node_Vx = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(x_in)
        node_Vx = BatchNormalization()(node_Vx)
        node_Vx = Activation('relu')(node_Vx)
        node_Vx = Dropout(dropout_value)(node_Vx)
        
        node_Vx = K.expand_dims(node_Vx, 1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        node_gateVx = Multiply()([edge_convnet_gate, node_Vx])  # B x V x V x H
        if aggregation=="mean":
            ReduceSum = Lambda(lambda z: K.sum(z, axis=2))
            node_gateVx_sum = ReduceSum(node_gateVx)
            edge_convnet_gate_sum = ReduceSum(edge_convnet_gate)
            
            divResult = Lambda(lambda x: x[0]/(x[1]+1e-20))
            mean_node = divResult([node_gateVx_sum,edge_convnet_gate_sum])
            node_convnet = Add()([node_Ux, mean_node])  # B x V x H
        elif aggregation=="sum":
            ReduceSum = Lambda(lambda z: K.sum(z, axis=2))
            node_gateVx_sum = ReduceSum(node_gateVx)
            node_convnet = Add()([node_Ux, node_gateVx_sum])  # B x V x H

        ## merge GNN layers
        node_combine.append(node_convnet)
        node_convnet = Concatenate()(node_combine)
        node_convnet = Conv1D(hidden_dim, kernel_size = filter_size, dilation_rate = d_rate, kernel_initializer = 'he_normal', padding = 'same')(node_convnet)

        ################# (1.4) Batch normalization for edge and node
        """Batch normalization for edge features.
        """
        """
        Args:
            e: Edge features (batch_size, num_nodes, num_nodes, hidden_dim)
    
        Returns:
            e_bn: Edge features after batch normalization (batch_size, num_nodes, num_nodes, hidden_dim)
        """
        """
        Args:
            x: Node features (batch_size, num_nodes, hidden_dim)
    
        Returns:
            x_bn: Node features after batch normalization (batch_size, num_nodes, hidden_dim)
        """
        # input: edge_convnet
        edge_convnet = BatchNormalization()(edge_convnet)
        node_convnet = BatchNormalization()(node_convnet)
    
        ################# (1.5) Relu Activation for edge and node
        edge_convnet = Activation('relu')(edge_convnet)
        node_convnet = Activation('relu')(node_convnet)
    
        ################# (1.6) Residual connection
        node_out = Add()([x_in, node_convnet])
        edge_out = Add()([e_in, edge_convnet])
    
        ################# (1.7) Update embedding
        nodes_embedding = node_out  # B x V x H
        edges_value_embedding = edge_out  # B x V x V x H
        
        d_rate = d_rate*2
        if d_rate > 4:
            d_rate = 4
    
    
    ################ (2) Define MLP classifiers for edge ################
    edge_merge_embedding = edges_value_embedding
    
    for i in range(num_lstm_layers):
        # LSTM
        tower_shape = K.int_shape(edge_merge_embedding)
        edge_merge_embedding = Lambda(ReshapeConv_to_LSTM)(edge_merge_embedding)
        tower_shape = K.int_shape(edge_merge_embedding)
        edge_merge_embedding = Bidirectional(ConvLSTM2D(filters=lstm_filter, kernel_size=(filter_size, filter_size),
                        input_shape=(None, tower_shape[1],tower_shape[2],tower_shape[-1]),
                        padding='same', return_sequences=True,  stateful = False), merge_mode='concat')(edge_merge_embedding)
        tower_shape = K.int_shape(edge_merge_embedding)
        LSTM_to_conv_dims = (tower_shape[1],tower_shape[2],tower_shape[-1])
        edge_merge_embedding = Lambda(ReshapeLSTM_to_Conv)(edge_merge_embedding)
        
    
    edge_merge_embedding = Activation('relu')(edge_merge_embedding)
    edge_merge_embedding = BatchNormalization()(edge_merge_embedding)
    edge_merge_embedding = Dropout(dropout_value)(edge_merge_embedding)
    edge_merge_embedding = Convolution2D(1, kernel_size = (filter_size, filter_size), dilation_rate=(d_rate, d_rate), padding = 'same')(edge_merge_embedding)
    
    nt_tower = Activation('sigmoid', name = "nt_out")(edge_merge_embedding)
    pair_tower2 = Activation('sigmoid', name = "pair_out2")(edge_merge_embedding) # subregion cross entropy
	
    edge_pred_model = Model([node_input, edges_value_input, edges_adj_input, edges_adj_input2], [pair_tower2,nt_tower])   
    return edge_pred_model

def weighted_binary_crossentropy_ntRegularized(y_true, y_pred) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    if not y_true.dtype == tf.float32:
        y_true = K.cast(y_true, tf.float32)
    if not y_pred.dtype == tf.float32:
        y_pred = K.cast(y_pred, tf.float32)
    logloss = -(1 - y_true) * K.log(1 - y_pred) 
    # Reduce the loss to a single scalar (sum or mean)
    return tf.reduce_mean(logloss)  # or tf.reduce_sum(weighted_bce)


def weighted_binary_crossentropy_pairRegularized(y_true, y_pred):
    masked = tf.not_equal(y_true,-1)
    masked2 = tf.equal(y_true,-1)
    zeros = tf.zeros_like(y_true)
    zeros = tf.cast(zeros, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred_sub = tf.where(masked, y_pred, zeros)
    y_true_sub = tf.where(masked, y_true, zeros)
    
    y_pred_rest = tf.where(masked2, y_pred, zeros)
    y_true_rest = tf.where(masked2, zeros, zeros)
    

    contact = tf.reduce_sum(K.cast(K.equal(y_true_sub, 1), tf.int32))/2
    non_contact = tf.reduce_sum(K.cast(K.equal(y_true_sub, 0), tf.int32))/2
    y_true_sub = K.clip(y_true_sub, K.epsilon(), 1-K.epsilon())
    y_pred_sub = K.clip(y_pred_sub, K.epsilon(), 1-K.epsilon())
    weight = contact * loss_ratio
    weight = K.clip(weight, 1, 100000)
    if not y_true_sub.dtype == tf.float32:
        y_true_sub = K.cast(y_true_sub, tf.float32)
    if not y_pred_sub.dtype == tf.float32:
        y_pred_sub = K.cast(y_pred_sub, tf.float32)
    
    if not y_true_rest.dtype == tf.float32:
        y_true_rest = K.cast(y_true_rest, tf.float32)
    if not y_pred_rest.dtype == tf.float32:
        y_pred_rest = K.cast(y_pred_rest, tf.float32)
    if not weight.dtype == tf.float32:
        weight = K.cast(weight, tf.float32)
    logloss = -(y_true_sub * K.log(y_pred_sub)* weight + (1 - y_true_sub) * K.log(1 - y_pred_sub)* weight/2) - ((1 - y_true_rest) * K.log(1 - y_pred_rest))
    # Reduce the loss to a single scalar (sum or mean)
    return tf.reduce_mean(logloss)  # or tf.reduce_sum(weighted_bce)


def train_graph_batched(model, epoch, data_generator, training=True, gcn_type=None):
    total_loss = 0
    total_batches = 0
    #print(f"Epoch: {epoch}")
    
    metrics_values = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mcc': []
    }
    if training:
        prefix = 'Train'
    else:
        prefix = 'Evaluation'

    zero_gradients = [tf.zeros_like(g) for g in model.trainable_variables]
    with tqdm(total=len(data_generator), desc=f"{prefix} Epoch {epoch}") as pbar:
        for graph_batch in data_generator:
            batch_loss = 0
            batch_gradients = []
            # Reset zero gradients at the start of each epoch or batch
            batch_gradients = [tf.identity(z) for z in zero_gradients]
            for graph in graph_batch:  
                
                #x, a, y = graph.x, graph.a, graph.y     
                X = tf.expand_dims(graph.x, axis=0) 
                EE_val = tf.expand_dims(graph.e, axis=0) 
                EE_adj = tf.expand_dims(graph.y[2], axis=0) 
                EE_adj2 = graph.a
                Y = tf.expand_dims(graph.y[0], axis=0) 
                Y_nt = tf.expand_dims(graph.y[1], axis=0) 
                if training:
                    with tf.GradientTape() as tape:
                        predictions = model([X,EE_val,EE_adj,EE_adj2], training=training)
                        y_pred = predictions[0]
                        y_nt_pred = predictions[1]
                        loss = lossWeights['pair_out2'] * (losses['pair_out2'](Y, y_pred)) + lossWeights['nt_out'] * (losses['nt_out'](Y_nt, y_nt_pred))
                        # Add regularization losses if any
                        loss += sum(model.losses)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    #if batch_gradients:
                    #    batch_gradients = [(accum_grad + grad) for accum_grad, grad in zip(batch_gradients, gradients)]
                    #else:
                    #    batch_gradients = gradients
                    
                    batch_gradients = [acc_g + g for acc_g, g in zip(batch_gradients, gradients)]
                    #optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                else:
                    predictions = model([X,EE_val,EE_adj,EE_adj2], training=training)
                    y_pred = predictions[0]
                    y_nt_pred = predictions[1]
                    loss = lossWeights['pair_out2'] * (losses['pair_out2'](Y, y_pred)) + lossWeights['nt_out'] * (losses['nt_out'](Y_nt, y_nt_pred))
                    # Add regularization losses if any
                    loss += sum(model.losses)
                batch_loss += loss.numpy()

                # Calculate other metrics
                prec = precision(Y, y_pred)
                rec = recall(Y, y_pred)
                f1_score = f1(Y, y_pred)
                mcc_score = mcc(Y, y_pred)
    
                # Update metric values list for averaging later
                metrics_values['precision'].append(prec.numpy())
                metrics_values['recall'].append(rec.numpy())
                metrics_values['f1'].append(f1_score.numpy())
                metrics_values['mcc'].append(mcc_score.numpy())
              
            # Apply gradients after processing the entire batch
            if training:
                optimizer.apply_gradients(zip(batch_gradients, model.trainable_variables))

            
            total_loss += batch_loss  # Ensuring scalar value of loss
            total_batches += len(graph_batch)

            avg_metrics = {key: np.mean(val) for key, val in metrics_values.items()}
            
                    
            # Update tqdm progress bar message
            pbar.set_postfix(batch=total_batches, loss=batch_loss, avg_loss=total_loss / total_batches, **avg_metrics)
            pbar.update(1)  # Manually increment the progress bar

    edge_pred_model = Model([node_input, edges_value_input, edges_adj_input, edges_adj_input2], [pair_tower2,nt_tower])   
    return edge_pred_model
