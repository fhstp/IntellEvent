import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.preprocessing.sequence import pad_sequences


def reshape_data(traj, grf):
    max_len = max([sequ.shape for sequ in grf])[0]
    rs_traj = [pad_sequences(sequ, maxlen=max_len, padding='post', dtype='float32') for sequ in traj]
    rs_traj = np.transpose(np.array(rs_traj).reshape(len(rs_traj), rs_traj[0].shape[0], max_len), (0, 2, 1))
    rs_grf = pad_sequences(np.array(grf), max_len, padding='post', dtype='int32')

    return rs_traj, rs_grf

##########################################################################################################################################
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
                        #if abs(distance) >= 5:
                        #    print(f"Label: {label}, GT [Frame]: {grf_events[l]}, Pred [Frame]: {loc[np.argmin(abs(loc - grf_events[l]))]}, "
                        #          f"Dist: {distance}, DBID: {is_dbid.iloc[idx_val]}, Trial: {is_trial.iloc[idx_val]}")
                                
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