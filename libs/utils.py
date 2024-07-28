import sys
from tensorflow.keras.utils import *
import tensorflow as tf
from tensorflow.keras import backend as K

#http://bioinformatics.hitsz.edu.cn/repRNA/static/download/physicochemical_property_indices.pdf
#	Adenine_content	 Cytosine content (3) 	Enthalpy (4) 	Enthalpy2 (4) 	Entropy (5) 	Entropy2 (5)	Free energy(5)	 Free energy2(5) 	GCcontent(3)	 Guaninecontent(3) Hy	drophilicity(2)	Hydrophilicity2	Keto (GT)content	Purine (AG)content	Rise (1)	Roll (1)	Shift (1)	Slide (1)	Stacking energy (1)	Thymine content (3)	Tilt (1)	Twist (1)

physicochemical_property_indices = {'AA':[2,0,-6.6,-6.82,-18.4,-19,-0.9,-0.93,0,0,0.023,0.04,0,2,3.18,7,-0.08,-1.27,-13.7,0,-0.8,31],
'AC':[1,1,-10.2,-11.4,-26.2,-29.5,-2.1,-2.24,1,0,0.083,0.14,0,1,3.24,4.8,0.23,-1.43,-13.8,0,0.8,32],
'AG':[1,0,-7.6,-10.48,-19.2,-27.1,-1.7,-2.08,1,1,0.035,0.08,0,2,3.3,8.5,-0.04,-1.5,-14,0,0.5,30],
'AU':[1,0,-5.7,-9.38,-15.5,-26.7,-0.9,-1.1,0,0,0.09,0.14,1,1,3.24,7.1,-0.06,-1.36,-15.4,1,1.1,33],
'CA':[1,1,-10.5,-10.44,-27.8,-26.9,-1.8,-2.11,1,0,0.118,0.21,0,1,3.09,9.9,0.11,-1.46,-14.4,0,1,31],
'CC':[0,2,-12.2,-13.39,-29.7,-32.7,-2.9,-3.26,2,0,0.349,0.49,0,0,3.32,8.7,-0.01,-1.78,-11.1,0,0.3,32],
'CG':[0,1,-8,-10.64,-19.4,-26.7,-2,-2.36,2,1,0.193,0.35,1,1,3.3,12.1,0.3,-1.89,-15.6,0,-0.1,27],
'CU':[0,1,-7.6,-10.48,-19.2,-27.1,-1.7,-2.08,1,0,0.378,0.52,1,0,3.3,8.5,-0.04,-1.5,-14,1,0.5,30],
'GA':[1,0,-13.3,-12.44,-35.5,-32.5,-2.3,-2.35,1,1,0.048,0.1,1,2,3.38,9.4,0.07,-1.7,-14.2,0,1.3,32],
'GC':[0,1,-14.2,-14.88,-34.9,-36.9,-3.4,-3.42,2,1,0.146,0.26,1,1,3.22,6.1,0.07,-1.39,-16.9,0,0,35],
'GG':[0,0,-12.2,-13.39,-29.7,-32.7,-2.9,-3.26,2,2,0.065,0.17,2,2,3.32,12.1,-0.01,-1.78,-11.1,0,0.3,32],
'GU':[0,0,-10.2,-11.4,-26.2,-29.5,-2.1,-2.24,1,1,0.16,0.27,2,1,3.24,4.8,0.23,-1.43,-13.8,1,0.8,32],
'UA':[1,0,-8.1,-7.69,-22.6,-20.5,-1.1,-1.33,0,0,0.112,0.21,1,1,3.26,10.7,-0.02,-1.45,-16,1,-0.2,32],
'UC':[0,1,-10.2,-12.44,-26.2,-32.5,-2.1,-2.35,1,0,0.359,0.48,1,0,3.38,9.4,0.07,-1.7,-14.2,1,1.3,32],
'UG':[0,0,-7.6,-10.44,-19.2,-26.9,-1.7,-2.11,1,1,0.224,0.34,1,1,3.09,9.9,0.11,-1.46,-14.4,1,1,31],
'UU':[0,0,-6.6,-6.82,-18.4,-19,-0.9,-0.93,0,0,0.389,0.44,2,0,3.18,7,-0.08,-1.27,-13.7,2,-0.8,31]}



def ReshapeConv_to_LSTM(x):
    reshape=K.expand_dims(x,0)
    return reshape

def ReshapeLSTM_to_Conv(x):
    reshape=K.squeeze(x,0)
    return reshape



          

# ------------- one hot encoding of RNA sequences -----------------#
def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in RNN_seq])
    # non standard nt is marked as [-1, -1, -1, -1]
    return feat

def l_mask(inp, seq_len):
    temp = []
    mask = np.ones((seq_len, seq_len))
    for k, K in enumerate(inp):
        if np.any(K == -1) == True:
            temp.append(k)
    mask[temp, :] = 0
    mask[:, temp] = 0
    return np.triu(mask, 2)



def output_mask(seq, NC=True):
    seq = seq.upper()
    if NC:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG', 'CC', 'GG', 'AG', 'CA', 'AC', 'UU', 'AA', 'CU', 'GA', 'UC']
    else:
        include_pairs = ['AU', 'UA', 'GC', 'CG', 'GU', 'UG']
    mask = np.zeros((len(seq), len(seq)))
    for i, I in enumerate(seq):
        for j, J in enumerate(seq):
            if str(I) + str(J) in include_pairs:
                mask[i, j] = 1
    return mask

def get_ss_pairs_from_matrix(pair_matrix,sequence,label_mask,Threshold):
    ones = np.ones((len(pair_matrix), len(pair_matrix)))
    #test_output= pair_matrix[np.triu(ones, 2) == 1][..., np.newaxis]
    test_output= np.triu(pair_matrix, 2)
    mask = output_mask(sequence, NC=flag_noncanonical)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    #for i in range(test_output.shape[0]):
    for i in range(len(inds[0])):
        y_pred[inds[0][i], inds[1][i]] = test_output[inds[0][i]][inds[1][i]]
    y_pred = np.multiply(y_pred, mask)
    
    tri_inds = np.triu_indices(y_pred.shape[0], k=1)

    out_pred = y_pred[tri_inds]
    outputs = out_pred[:, None]
    
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([sequence[tri_inds[0][j]], sequence[tri_inds[1][j]]])] for j in
                 range(tri_inds[0].shape[0])]
    #print(seq_pairs)
    outputs_T = np.greater_equal(outputs, Threshold)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    pred_pairs = [i[:2] for i in pred_pairs]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred) # remove temporary
    pred_pairs_pred = [str(i[0]+1)+'-'+str(i[1]+1) for i in pred_pairs]
    return pred_pairs_pred


# ----------------------- find multiplets pairs--------------------------------#
def multiplets_pairs(pred_pairs):

    pred_pair = [i[:2] for i in pred_pairs]
    temp_list = flatten(pred_pair)
    temp_list.sort()
    new_list = sorted(set(temp_list))
    dup_list = []
    for i in range(len(new_list)):
        if (temp_list.count(new_list[i]) > 1):
            dup_list.append(new_list[i])

    dub_pairs = []
    for e in pred_pair:
        if e[0] in dup_list:
            dub_pairs.append(e)
        elif e[1] in dup_list:
            dub_pairs.append(e)

    temp3 = []
    for i in dup_list:
        temp4 = []
        for k in dub_pairs:
            if i in k:
                temp4.append(k)
        temp3.append(temp4)
        
    return temp3

def multiplets_free_bp(pred_pairs, y_pred):
    L = len(pred_pairs)
    multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = []
    while len(multiplets_bp) > 0:
        remove_pairs = []
        for i in multiplets_bp:
            save_prob = []
            for j in i:
                save_prob.append(y_pred[j[0], j[1]])
            remove_pairs.append(i[save_prob.index(min(save_prob))])
            save_multiplets.append(i[save_prob.index(min(save_prob))])
        pred_pairs = [k for k in pred_pairs if k not in remove_pairs]
        multiplets_bp = multiplets_pairs(pred_pairs)
    save_multiplets = [list(x) for x in set(tuple(x) for x in save_multiplets)]
    assert L == len(pred_pairs)+len(save_multiplets)
    #print(L, len(pred_pairs), save_multiplets)
    return pred_pairs, save_multiplets



def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def output_result_simple(avg_acc):
    if len(avg_acc) == 0:
        avg_acc = [0,0,0,0,0,0,0]
    print("Range(> 2):")
    print("Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy")
    print("Acc:     %.3f        %.3f        %.3f      %.3f      %.3f      %.3f      %.3f" \
            %(avg_acc[0], avg_acc[1], avg_acc[2], avg_acc[3],avg_acc[4],avg_acc[5],avg_acc[6]))
    

def ct_file_output(pairs, seq, save_result_path):

    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)
    for i, I in enumerate(pairs):
        arr = I.split('-')
        col5[int(arr[0])-1] = int(arr[1])
        col5[int(arr[1])-1] = int(arr[0])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
                      
    np.savetxt(save_result_path, (temp), delimiter='\t', fmt="%s", header=str(len(seq)) + '\t\t' + 'Prediction' + '\t\t' + 'output' , comments='')

    return

  
    avg_loss = total_loss / total_batches
    return model, total_loss / total_batches, avg_metrics['f1']
