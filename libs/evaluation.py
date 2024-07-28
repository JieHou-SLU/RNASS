import tensorflow as tf
from libs.utils import *

### evaluation

def evaluate_predictions_single(reference_matrix,pred_matrix,sequence,label_mask, verbose=False):
    #print("reference_structure.shape: ",reference_matrix.shape)
    #print("predicted_structure.shape: ",pred_matrix.shape)
    reference_structure = get_ss_pairs_from_matrix(reference_matrix,sequence,label_mask, 0.5)
    predicted_structure = get_ss_pairs_from_matrix(pred_matrix,sequence,label_mask, 0.5)
    if verbose:
      print("reference_structure len: ",len(reference_structure))
      print("reference_structure: ",reference_structure)
      print("predicted_structure len: ",len(predicted_structure))
      print("predicted_structure: ",predicted_structure)
    acc = compare_structures('N'*len(reference_matrix),reference_structure,predicted_structure)
    
    return acc


# Compare predicted structure against ref. Calculate similarity statistics
def compare_structures(ref_seq,pred, ref, verbose = 0):
    pred_p = pred_n = tp = tn = 0
    # Count up how many SS and DS predictions were correct, relative to the phylo structure
    n_pairs = len(ref_seq)
    for i in range(1, n_pairs+1):
        for j in range(i+1, n_pairs+1):
            pair = str(i)+'-'+str(j)
            if pair in pred:
                pred_p += 1
                if pair in ref:
                    tp += 1
            else:
                pred_n += 1
                if pair not in ref:
                    tn += 1
    fp = pred_p - tp
    fn = pred_n - tn
    fp = float(fp)
    tp = float(tp)
    fn = float(fn)
    fp = float(fp)
    ppv 		= round((100 * tp) / (tp + fp+ sys.float_info.epsilon), 2)
    sensitivity	= round((100 * tp) / (tp + fn+ sys.float_info.epsilon), 2)
    npv			= round((100 * tn) / (tn + fn+ sys.float_info.epsilon), 2)
    specificity	= round((100 * tn) / (tn + fp+ sys.float_info.epsilon), 2)
    accuracy 	= round((100 * (tp + tn)) / (tp + fp + tn + fn+ sys.float_info.epsilon), 2)
    
    f1_score = round((2 * ppv * sensitivity) / (ppv + sensitivity+ sys.float_info.epsilon),2)
    up = tp*tn - fp*fn
    down = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    
    mcc = up / (down + sys.float_info.epsilon)*100
    mcc = round(mcc,2)
    if verbose != 0:
        print("Method    ppv         sensitivity          npv        specificity        f1_score        mcc        accuracy")
        print("Results with respect to SS :     %.3f        %.3f        %.3f      %.3f      %.3f      %.3f      %.3f" \
            %(ppv, sensitivity, npv, specificity, f1_score, mcc,accuracy))

    return [ppv, sensitivity, npv, specificity, f1_score, mcc,accuracy]


def precision(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def specificity(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def negative_predictive_value(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())


def f1(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def fbeta(y_true, y_pred, beta=2, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)

#matthews_correlation_coefficient
def mcc(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())


def equal_error_rate(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    n_imp = tf.math.count_nonzero(tf.equal(y_true, 0), dtype=tf.float32) + tf.constant(K.epsilon())
    n_gen = tf.math.count_nonzero(tf.equal(y_true, 1), dtype=tf.float32) + tf.constant(K.epsilon())

    scores_imp = tf.boolean_mask(y_pred, tf.equal(y_true, 0))
    scores_gen = tf.boolean_mask(y_pred, tf.equal(y_true, 1))

    loop_vars = (tf.constant(0.0), tf.constant(1.0), tf.constant(0.0))
    cond = lambda t, fpr, fnr: tf.greater_equal(fpr, fnr)
    body = lambda t, fpr, fnr: (
        t + 0.001,
        tf.divide(tf.math.count_nonzero(tf.greater_equal(scores_imp, t), dtype=tf.float32), n_imp),
        tf.divide(tf.math.count_nonzero(tf.less(scores_gen, t), dtype=tf.float32), n_gen)
    )
    t, fpr, fnr = tf.while_loop(cond, body, loop_vars, back_prop=False)
    eer = (fpr + fnr) / 2

    return eer


#https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
def recall(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras

def tp(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    #y_true = tf.reshape(y_true, [-1])
    #y_pred = tf.reshape(y_pred, [-1])
    #y_true_shape = K.print_tensor(y_true.get_shape(), message='y_true = ')
    #y_pred_shape = K.print_tensor(y_pred.get_shape(), message='y_pred = ')
    true_positives = tf.reduce_sum(tf.cast(tf.greater(y_true * y_pred, 0.5), tf.int32))/2
    possible_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.int32))/2
    return true_positives

def pp(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    #y_true = tf.reshape(y_true, [-1])
    #y_pred = tf.reshape(y_pred, [-1])
    true_positives = tf.reduce_sum(tf.cast(tf.greater(y_true * y_pred, 0.5), tf.int32))/2
    possible_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), tf.int32))/2
    return possible_positives

def num_contact(y_true, y_pred, threshold=0.5):
    y_pred = tf.cast(y_pred > threshold, tf.float32)  # Apply threshold
    contact = tf.reduce_sum(K.cast(K.equal(y_true, 1), tf.int32))/2
    non_contact = tf.reduce_sum(K.cast(K.equal(y_true, 0), tf.int32))/2
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    weight = contact
    if not weight.dtype == tf.float32:
        weight = K.cast(weight, tf.float32)
    return weight

def data_size(y_true, y_pred, threshold=0.5):
    size =K.shape(y_true)
    return size
    
          