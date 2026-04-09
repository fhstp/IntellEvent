import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_absolute_error
from scipy.signal import find_peaks
from tensorflow import keras

##########################################################################################################################################
# Function: 'deterministic_ops'

    # Purpose:
        # Sets all seeds and environment variables to ensure reproducible results across different runs.

    # Input Parameters:
        # seed: An integer value used to initialize the random number generators for Python, NumPy, and TensorFlow.

    # Output:
        # None

    # Description:
        # The function sets the 'PYTHONHASHSEED', seeds the random and numpy libraries, and configures TensorFlow 
        # to use deterministic operations. This is crucial for debugging and comparing model architectures accurately.
##########################################################################################################################################
def deterministic_ops(seed=21):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    tf.config.experimental.enable_op_determinism()

##########################################################################################################################################
# Function: 'get_train_test_split'

    # Purpose:
        # This function is used to split a dataset into train, test, and validation sets. 

    # Input Parameters:
        # data: This parameter is a pandas DataFrame that contains the dataset to be split. 
        # The dataset needs to have 'Label' and 'DBid' columns to split by pathology and maintain subject stratification.

        # hold_out: A float (0-1) specifying the proportion of unique subjects (DBid) to assign to the test set.

    # Output:
        # tr_data: DataFrame containing the training data.
        # te_data: DataFrame containing the test data.
        # tr_val_data: DataFrame containing the stratified training portion for validation.
        # te_val_data: DataFrame containing the stratified test portion for validation.

    # Description:
        # The function identifies unique labels and splits the 'DBid's for each label to ensure the test set 
        # is representative of all pathologies. It then performs a secondary split on the training data 
        # to create a validation set (fixed at 10% of the training data).
##########################################################################################################################################
def get_train_test_split(data, hold_out=0.3):
    labels = np.unique(data['Label'])
    tr_data_list, te_data_list = [], []
    
    for label in labels:
        tmp_data = data[data['Label'] == label]
        unique_ids = tmp_data['DBid'].unique()
        tr_ids, te_ids = train_test_split(unique_ids, test_size=hold_out, random_state=0)
        tr_data_list.append(tmp_data[tmp_data['DBid'].isin(tr_ids)])
        te_data_list.append(tmp_data[tmp_data['DBid'].isin(te_ids)])
    
    tr_data = pd.concat(tr_data_list, ignore_index=True)
    te_data = pd.concat(te_data_list, ignore_index=True)
    
    tr_val_list, te_val_list = [], []
    for label in labels:
        tmp_data = tr_data[tr_data['Label'] == label]
        tr_v_ids, te_v_ids = train_test_split(tmp_data['DBid'].unique(), test_size=0.1, random_state=0)
        tr_val_list.append(tmp_data[tmp_data['DBid'].isin(tr_v_ids)])
        te_val_list.append(tmp_data[tmp_data['DBid'].isin(te_v_ids)])
        
    return tr_data, te_data, pd.concat(tr_val_list, ignore_index=True), pd.concat(te_val_list, ignore_index=True)

##########################################################################################################################################
# Function: 'event_seperator'

    # Purpose:
        # Trims and re-labels gait sequences based on Initial Contact (IC) or Foot Off (FO) events.

    # Input Parameters:
        # tr_grf: Series containing Ground Reaction Force event labels.
        # tr_traj: Series containing trajectory/velocity data.
        # event_type: String ('IC' or 'FO') to determine the re-labeling logic.

    # Output:
        # tr_grf: Processed and trimmed event labels.
        # tr_traj: Processed and trimmed trajectory sequences.

    # Description:
        # It maps specific gait phases to binary labels (0 or 1). It then adds random padding (5-125 frames) 
        # around the first and last detected events to create robust training samples.
##########################################################################################################################################
def event_seperator(tr_grf, tr_traj, event_type=True):
    for idx in range(len(tr_grf)):
        grf = tr_grf.iloc[idx].copy()
        traj = tr_traj.iloc[idx].copy()

        if event_type == True:
            grf = np.where(grf == 3, 1, np.where((grf == 2) | (grf == 4), 0, grf))
        else:
            grf = np.where((grf == 2) | (grf == 4), 1, np.where((grf == 1) | (grf == 3), 0, grf))

        loc = np.argwhere(grf > 0)
        if len(loc) == 0: continue

        first_idx, last_idx = loc[0][0], loc[-1][0]
        rand_len_fore = random.randint(5, 125) if first_idx >= 140 else random.randint(5, max(1, first_idx-1))
        rand_len_aft = random.randint(5, 125) if len(grf) - last_idx >= 140 else random.randint(5, max(1, len(grf)-last_idx-1))

        start, end = max(0, first_idx - rand_len_fore), min(len(grf), last_idx + rand_len_aft)
        tr_grf.iloc[idx], tr_traj.iloc[idx] = grf[start:end], traj[:, start:end]
            
    return tr_grf, tr_traj

##########################################################################################################################################
# Function: 'reshape_data'

    # Purpose:
        # Pads variable-length sequences to a uniform length for batch processing.

    # Input Parameters:
        # traj: List/Series of trajectory sequences.
        # grf: List/Series of event sequences.

    # Output:
        # rs_traj: Numpy array of padded trajectory data.
        # rs_grf: Numpy array of padded event data.

    # Description:
        # It finds the maximum sequence length in the batch and applies post-padding using zero values. 
        # The trajectory data is transposed to ensure the dimensions align for RNN input (Samples, Time, Features).
##########################################################################################################################################
def reshape_data(traj, grf):
    max_len = max([sequ.shape[0] for sequ in grf])
    rs_traj = [pad_sequences(sequ, maxlen=max_len, padding='post', dtype='float32') for sequ in traj]
    rs_traj = np.transpose(np.array(rs_traj).reshape(len(rs_traj), traj.iloc[0].shape[0], max_len), (0, 2, 1))
    rs_grf = pad_sequences(grf, maxlen=max_len, padding='post', dtype='float32')
    return rs_traj, rs_grf

def get_sample_weights(y, weights):
    """Assigns weights to classes to handle imbalance."""
    return np.where(y == 0, weights[0], weights[1])


class CheckMAE(keras.callbacks.Callback):
    """Callback to monitor Mean Absolute Error per pathology class during training."""
    def __init__(self, x_test, y_test,labels, is_IC):
        self.x_test = x_test
        self.y_test = y_test
        self.labels = labels
        self.is_IC = is_IC
        self.class_ids = sorted(set(labels))
        
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test, verbose=0)
        mae_list = eval_val_data(self.labels, y_pred, self.y_test, self.is_IC)

        mae_dict = {c: mae_list[i] for i, c in enumerate(self.class_ids)}
        print(f"\nEpoch {epoch + 1} - MAEs by class: " +
              " | ".join([f"Class {c}: {mae:.4f}" for c, mae in mae_dict.items()]))
        logs["val_mae"] = np.mean(mae_list) 



##########################################################################################################################################
# Function: 'eval_te_data'

    # Purpose:
        # Evaluates model performance on the test dataset by calculating Mean Absolute Error (MAE) 
        # and detection accuracy (TP vs FN) for gait events.

    # Input Parameters:
        # labels: Series or array containing the pathology labels for the test samples.
        # ids: Series or array of subject IDs (DBid) for the test samples.
        # trials: Series or array containing the trial identifiers.
        # preds: Numpy array of model predictions (probabilities between 0 and 1).
        # te_grf: Numpy array or list of the ground truth event sequences.
        # is_IC: Boolean flag. If True, evaluates Initial Contact (IC); if False, evaluates Foot Off (FO).

    # Output:
        # ic_mae_all: A list containing the Mean Absolute Error for each pathology label.
        # ic_list_all: A nested list of all frame distances between predicted and ground truth events.
        # ic_tp_list: List of True Positive counts per label (distance < 4 frames).
        # ic_fn_list: List of False Negative counts per label (distance >= 4 frames or no event predicted).

    # Description:
        # This function iterates through each pathology label and processes each prediction sequence.
        # It uses scipy's find_peaks to identify predicted events and compares them to ground truth events.
        # An event is classified as a True Positive (TP) only if the distance to the nearest ground truth 
        # is less than 4 frames. All other cases—where the distance is 4 or greater, or where no peak 
        # is detected—are classified as False Negatives (FN). The function also prints details for 
        # any event with a distance >= 5 frames for outlier investigation.
##########################################################################################################################################
def eval_te_data(labels, ids, trials, preds, te_grf, is_IC):
    # Initialize metrics and outputs
    ic_list_all, ic_tp_list, ic_fn_list, ic_mae_all = [], [], [], []
    unq_labels = np.unique(labels)
    
    # THRESHOLD PARAMETERS
    min_peak_threshold = 0.1
    peak_distance = 20
    ground_truth_threshold = 4
    
    for label in unq_labels:
        ic_all, ic_tp, ic_fn = [], 0, 0
        
        # Filter data for the current label
        mask = (labels == label)
        is_preds = preds[mask]
        is_grf = te_grf[mask]
        is_dbid = ids[mask]
        is_trial = trials[mask]

        for idx_val in range(is_preds.shape[0]):
            pred = is_preds[idx_val] 
            
            # Ground truth events: IC (1, 3) or FO (2, 4)
            if is_IC:
                grf_events = np.append(np.where(is_grf[idx_val] == 1)[0], np.where(is_grf[idx_val] == 3)[0])
            else:
                grf_events = np.append(np.where(is_grf[idx_val] == 2)[0], np.where(is_grf[idx_val] == 4)[0])

            # Detect peaks in model output
            [loc, _] = find_peaks(pred[:, 0], height=min_peak_threshold, distance=peak_distance)

            for l in range(grf_events.shape[0]):
                # If no peaks are found -> False Negative
                if not loc.any():
                    print(f"False Negative - No Event Predicted | Label: {label}, DBID: {is_dbid.iloc[idx_val]}, Trial: {is_trial.iloc[idx_val]}")
                    ic_fn += 1
                else:
                    # Find distance to the closest predicted peak
                    distance = loc[np.argmin(abs(loc - grf_events[l]))] - grf_events[l]
    
                    # TP: distance < +/- 4
                    if abs(distance) < ground_truth_threshold:
                        ic_all.append(distance)
                        ic_tp += 1
                    # FN: Everything else (distance >= 4)
                    else:
                        ic_all.append(distance)
                        ic_fn += 1
                        
                        # Outlier detector for errors >= 5 frames
                        if abs(distance) >= 5:
                            print(f"Label: {label}, GT [Frame]: {grf_events[l]}, Pred [Frame]: {loc[np.argmin(abs(loc - grf_events[l]))]}, "
                                  f"Dist: {distance}, DBID: {is_dbid.iloc[idx_val]}, Trial: {is_trial.iloc[idx_val]}")
                                
        # Calculate MAE per label
        if len(ic_all) > 0:                          
            ic_mae_all.append(mean_absolute_error(np.zeros(len(ic_all)), ic_all))
        else:
            ic_mae_all.append(999.99999)
            
        ic_list_all.append(ic_all)
        ic_tp_list.append(ic_tp)
        ic_fn_list.append(ic_fn)
        
    return ic_mae_all, ic_list_all, ic_tp_list, ic_fn_list

##########################################################################################################################################
# Function: 'eval_val_data'

    # Purpose:
        # A lightweight evaluation function used during training/validation to monitor MAE and Precision.

    # Input Parameters:
        # labels: Series or array of pathology labels for the validation set.
        # preds: Numpy array of model predictions.
        # te_grf: Numpy array of ground truth event sequences.
        # is_IC: Boolean flag for Initial Contact (True) or Foot Off (False).

    # Output:
        # ic_mae_all: A list of Mean Absolute Errors calculated per pathology label.
        # precision: A single float representing the global precision across all samples in the validation set.

    # Description:
        # This function functions similarly to eval_te_data but is optimized for speed during the validation phase.
        # It calculates the Precision and F1-score (calculated but not returned) to provide a quick snapshot 
        # of model accuracy during hyperparameter tuning or training epochs.
##########################################################################################################################################
def eval_val_data(labels, preds, te_grf, is_IC):
    ic_tp_list, ic_fn_list, ic_mae_all = [], [], []
    unq_labels = np.unique(labels)
    # Removed fp_window as it is no longer used
    min_peak_threshold, peak_distance, ground_truth_threshold = 0.1, 20, 4
    
    for label in unq_labels:
        ic_all, ic_tp, ic_fn = [], 0, 0
        mask = (labels == label)
        is_preds, is_grf = preds[mask], te_grf[mask]

        for idx_val in range(is_preds.shape[0]):
            pred = is_preds[idx_val] 
            # Identify ground truth events based on IC or FO type
            grf_events = np.append(np.where(is_grf[idx_val] == (1 if is_IC else 2))[0], 
                                   np.where(is_grf[idx_val] == (3 if is_IC else 4))[0])

            [loc, _] = find_peaks(pred[:,0], height=min_peak_threshold, distance=peak_distance)

            for l in range(grf_events.shape[0]):
                # If no peaks are found, it's a False Negative
                if not loc.any():
                    ic_fn += 1
                else:
                    # Calculate distance to the nearest predicted peak
                    distance = loc[np.argmin(abs(loc - grf_events[l]))] - grf_events[l]
                    
                    # TP only if distance is within the threshold (< 4)
                    if abs(distance) < ground_truth_threshold:
                        ic_all.append(distance)
                        ic_tp += 1
                    # Everything else is a False Negative
                    else:
                        ic_all.append(distance)
                        ic_fn += 1
                                
        ic_mae_all.append(mean_absolute_error(np.zeros(len(ic_all)), ic_all) if ic_all else [999.99999])
        ic_tp_list.append(ic_tp)
        ic_fn_list.append(ic_fn)

    tp_total = sum(ic_tp_list)
    
    return ic_mae_all