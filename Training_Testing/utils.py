import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, LSTM, BatchNormalization
from tensorflow.keras import layers, regularizers
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

##########################################################################################################################################
# Function: 'get_train_test_split'

    # Purpose:
        # This function is used to split a dataset into train, test, and validation sets. 
        # It splits the data based on the labels present in the dataset and assigns a certain proportion of data to each set.

    #Input Parameters:
        # data: This parameter is a pandas DataFrame that contains the dataset to be split. 
        # The dataset should needs to have these two columns in order to split by pathology (label)
        # as well as stratified (DBid): label and DBid.

        # hold_out: This parameter is a float value between 0 and 1 that specifies the proportion of data to be assigned to the test set.
        # The remaining data is assigned to the train set.

        # validation: This parameter is a float value between 0 and 1 that specifies the proportion of data to be 
        # assigned to the validation set.

    # Output:
        # tr_data: This DataFrame contains the training data.
        # te_data: This DataFrame contains the test data.
        # tr_val_data: This DataFrame contains the training data for the validation set.
        # te_val_data: This DataFrame contains the test data for the validation set.

    # Description:
        # The get_train_test_split() function first identifies the unique labels present in the input dataset. 
        # For each label, the function splits the data into training and test sets based on the hold_out parameter 
        # using the train_test_split() function from the scikit-learn library. It assigns the training data 
        # to tr_data and the test data to te_data.
        # If the label is equal to 2, all data with that label is assigned to the test set to ensure that patients 
        # with a certain condition are not split between the training and test sets. (In our case this was for drop foot
        # patients due to a low number). Remove this if not needed.
        # The function then splits the training data into training and validation sets based on the validation parameter 
        # using the train_test_split() function. It assigns the training data for the validation set to tr_val_data and 
        # the test data for the validation set to te_val_data.


def get_train_test_split(data, hold_out, validation):
    # Train - Test Split
    labels = np.unique(data['label'])
    tr_data = pd.DataFrame()
    te_data = pd.DataFrame()

    for i in range(0, len(labels)):
        is_label = data['label'] == labels[i]
        tmp_data = data[is_label]

        if i != 2:
            tr_DBid, te_DBid = train_test_split(np.unique(tmp_data['DBid']), test_size=hold_out, random_state=0)
            is_train = np.in1d(tmp_data['DBid'], tr_DBid)
            is_test = np.in1d(tmp_data['DBid'], te_DBid)

            tr_data = tr_data.append(tmp_data[is_train], ignore_index=True)
            te_data = te_data.append(tmp_data[is_test], ignore_index=True)

        if i == 2:  # append all ICP - Spitzfuß patients to the TE_DATA
            te_data = te_data.append(tmp_data)


    # Validation Data
    tr_val_data = pd.DataFrame()
    te_val_data = pd.DataFrame()

    labels = [1, 2, 5, 6]
    for i in range(0, len(labels)):
        is_label = tr_data['label'] == labels[i]
        tmp_data = tr_data[is_label]

        tr_val_DBid, te_val_DBid = train_test_split(np.unique(tmp_data['DBid']), test_size=validation, random_state=0)
        is_train = np.in1d(tmp_data['DBid'], tr_val_DBid)
        is_test = np.in1d(tmp_data['DBid'], te_val_DBid)

        tr_val_data = tr_val_data.append(tmp_data[is_train], ignore_index=True)
        te_val_data = te_val_data.append(tmp_data[is_test], ignore_index=True)


    return tr_data, te_data, tr_val_data, te_val_data
##########################################################################################################################################

##########################################################################################################################################
# Function: 'reshape_data':

    #Purpose:
        # This function reshapes and pads sequences of trajectory and GRF data to the longest input sequence,
        # so that they can be used as input for the neural network. The input shape of tr_traj is (features, num_frames).
        # Specifically, it pads the sequences with zeros so that all sequences have the same length 
        # (the length of the longest sequence), and it reshapes the data to have the shape 
        # (num_samples, max_seq_length, num_features), as this is the input format for the neural network.  
        
    # Inputs:
        # traj: A list of numpy arrays where each array contains a sequence of data points (features, num_frames)
        # representing the trajectories. Each array can have a different number of time steps.
        
        # grf: A list of numpy arrays where each array contains a sequence of data points representing 
        # the ground reaction forces for the same trial as the trajectory data. Each array can have a different number of time steps.

    # Outputs:
        # rs_traj: A numpy array with shape (num_sequences, max_len, num_features) where num_sequences is the number 
        # of sequences in traj, max_len is the maximum length of the sequences in traj, and num_features is the number 
        # of features in each time step of the sequences in traj. The rs_traj array is obtained by padding the sequences 
        # in traj with zeros to make them all the same length and then transposing the resulting array.
        
        # rs_grf: A numpy array with shape (num_sequences, max_len) where num_sequences is the number of sequences
        # in grf and max_len is the maximum length of the sequences in grf. The rs_grf array is obtained by padding 
        # the sequences in grf with zeros to make them all the same length.
        
    # Description:
        # The reshape_data function takes as input two lists of numpy arrays, traj and grf. 
        # The function first determines the maximum length of the sequences in traj. 
        # It then pads (right pad) each sequence in traj with zeros to make them all the same length 
        # using the pad_sequences function from the tensorflow.keras.preprocessing.sequence module. 
        # The resulting sequences are then transposed to create a numpy array with shape (num_sequences, max_len, num_features)
        # where num_sequences is the number of sequences in traj, max_len is 
        # the maximum length of the sequences in traj, and num_features is the number of features in each time step of the 
        # sequences in traj.
        # The function also pads each sequence in grf with zeros to make them all the same length using 
        # the pad_sequences function. The resulting numpy array has shape (num_sequences, max_len).

def reshape_data(traj, grf):
    max_len = max([sequ.shape[1] for sequ in traj])
    rs_traj = [pad_sequences(sequ, maxlen=max_len, padding='post', dtype='float32') for sequ in traj]
    rs_traj = np.transpose(np.array(rs_traj).reshape(len(rs_traj), rs_traj[0].shape[0], max_len), (0, 2, 1))
    rs_grf = pad_sequences(np.array(grf), max_len, padding='post', dtype='int32')
    
    return rs_traj, rs_grf

##########################################################################################################################################

##########################################################################################################################################


# Function: 'get_sample_weights'

    # Purpose: 
        # The purpose of this function is to assign a weight to each sample in a dataset (for the grf ground truth)
        # based on its class label. It takes a numpy array of samples and a list of weights corresponding to each class label,
        # and returns a numpy array of weights for each sample. 
        # This function can be useful in scenarios where the class distribution in the dataset is imbalanced, 
        # and assigning weights to each sample can help to balance the contribution of each class to the overall model performance.
        # Especially, when the ratio of non-event to event is  ~ 1:200 like in the case of gait events. 

    # Inputs:
        # samples (numpy array): A numpy array containing the samples. It should be a 2-dimensional array, 
        # where each row represents a sample and each column represents a feature.
        
        # weight (list): A list of weights corresponding to each class label. 
        # The weight for each class should be a positive floating-point number.
        
    # Returns:
        # samples (numpy array): A numpy array containing the weights for each sample. It has the same shape as the samples input array.
        
    # Functionality:
        #The function assigns a weight to each sample based on its class label.  
        # The weights are determined by the weight list, where each element of the list corresponds to a class label.
        # For example, if there are three class labels, the weight list should have three elements, 
        # and the first element of the list corresponds to the weight for the first class, 
        # the second element of the list corresponds to the weight for the second class, and so on.

        # The function replaces the class label of each sample with the corresponding weight from the weight list. 
        # This is done using a loop that iterates over the length of the weight list. 
        # The samples array is first cast to a float32 data type to ensure that the weights can be assigned as floating-point numbers.
        # After assigning the weights, the samples array is reshaped to its original shape and returned.

def get_sample_weights(samples, weight):
    samples = np.float32(samples)
    for i in range(0, len(weight)):
        samples[samples == i] = weight[i]
    samples = samples.reshape(len(samples), samples.shape[1])
    return samples

##########################################################################################################################################

##########################################################################################################################################
# Function: 'get_uncompiled_lstm_model'

    #Purpose:
        # The purpose of this function is to return an uncompiled LSTM model with the specified architecture.
        # The model includes masking of zero values using `Masking` layer, bidirectional LSTMs with dropout layers, 
        # and a `TimeDistributed` dense layer with a softmax activation function. 

    # Inputs:
        # features (int): The number of input features. 
        # depth (int): The number of hidden units in each LSTM layer.
        # dropout_val (float): The dropout rate between LSTM layers.
        # layer_size (int): The number of LSTM layers in the model.
        #  dense_out (int): The number of output classes.
        
    # Returns:
        # keras.Sequential: An uncompiled LSTM model with the specified architecture.
        
    # Functionality:
        # This function creates a sequential model using Tensorflow Keras API, and adds layers to it based on the input parameters. 
        # It first applies masking to the input sequence data using the `Masking` layer. 
        # It then adds a specified number of bidirectional LSTM layers, each with the specified number of hidden units and dropout rate. 
        # Finally, it adds a `TimeDistributed` dense layer with a softmax activation function to generate predictions for each time step.
        # The output is an uncompiled Keras model with the specified architecture, which can be compiled and trained with the 
        # `compile` and `fit` methods.

def get_uncompiled_lstm_model(features, depth, dropout_val, layer_size, dense_out):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0.))
    for i in range(0, layer_size):
        model.add(Bidirectional(LSTM(depth, return_sequences=True, input_shape=(None, features)), merge_mode='concat'))
        model.add(layers.Dropout(dropout_val))

    model.add(layers.TimeDistributed(layers.Dense(dense_out, activation='softmax')))
    return model

##########################################################################################################################################

##########################################################################################################################################
# Function: 'get_compiled_model'

    # Purpose:
        # This function returns a compiled LSTM model that can be used for sequence classification tasks. 
        # The model consists of multiple LSTM layers, with optional dropout regularization, 
        # followed by a output layer with a softmax activation function. 
        # The compiled model is ready to be trained on a dataset and used for prediction.

    # Inputs:
        # features (int): The number of features.
        # depth (int): The number of LSTM hidden units in each layer.
        # dropout_val (float): The dropout rate, between 0 and 1, for the LSTM layers. 
        # layer_size (int): The number of LSTM layers in the model.
        # dense_out (int): The number of output classes. 
    
    # Returns:
        # model (tf.keras.Model): A compiled LSTM model, ready to be trained on a dataset and used for prediction.

    # Functionality:
        # The function first calls the get_uncompiled_lstm_model function, which returns an uncompiled LSTM model 
        # with the specified number of layers, units, dropout rate, and output classes.

        # Then, the function compiles the model using the CategoricalCrossentropy loss function, 
        # the Adam optimizer, and the mean_absolute_error metric for evaluation. 
        # Finally, the function returns the compiled model for use in training and prediction.

def get_compiled_model(features, depth, dropout_val, layer_size, dense_out):
    model = get_uncompiled_lstm_model(features, depth, dropout_val, layer_size, dense_out)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),  
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision() ] 
    )  
    return model

##########################################################################################################################################

##########################################################################################################################################

# Function: ' eval_fc_te_data:

    # Purpose: 
        # The function evaluates the performance of the input model on the test dataset. 
        #  It calculates various metricss for the unique labels in the labels array (for different pathologies separately).

    # Inputs:
        # model: a TensorFlow Keras model
        # te_traj: a numpy array of shape (num_samples, num_frames, num_features) containing the test trajectory data
        # te_grf: a numpy array of shape (num_samples, num_frames) containing the GRF data for the test trials
        # labels: a numpy array of shape (n_samples,) containing integer labels for each trial in te_traj to identify 
        # results for different pathologies
        
    # The function sets two threshold parameters:
        # min_peak_threshold: a float representing the minimum peak threshold for detecting GRF events
        # ground_truth_threshold: an integer representing the maximum allowed distance between predicted and ground truth event frames
    
    # Returns:
        # the mean absolute error over all true positive and false positive events
        # the list of all events as a distance to the ground truth events (0 = zero frames off, 1 = one frame off, and so on)
        # the classifications (true positive, false positive and false negative)
        # In a list segmented for each unique label (= each pathology)
 
    # Functionality:
        # The function then loops over each trial in the test dataset, and for each trial, loops over each output channel 
        # of the prediction. For each output channel, the function finds all peaks in the corresponding prediction 
        # with a height greater than 'min_peak_threshold'. It then compares the detected peaks to the ground truth event
        # frames too the corresponding GRF events.

        # If no peak is detected, the function considers this a false negative event. 
        # If at least one peak is detected, the function calculates the distance between each detected peak and the nearest ground
        # truth event frame. If the distance is less than or equal to 'ground_truth_threshold', the event is considered a true positive.
        # If the distance is greater than ground_truth_threshold but less than or equal to 50 frames, the event is 
        # considered a false positive. If the distance is greater than 50 frames, the event is considered a false negative.
        
def eval_fc_te_data(model, te_traj, te_grf, labels):
    fc_list_all = list()
    fc_mae = list()
    fc_tp = list()
    fc_fp = list()
    fc_fn = list()
    
    unq_labels = np.unique(labels)
    # THRESHOLD PARAMETERS
    min_peak_threshold = 0.01
    ground_truth_threshold = 5

    # Predict Test-Data
    te_traj = [traj.reshape(1, te_traj.shape[1], te_traj.shape[2]) for traj in te_traj]
    te_traj = tf.data.Dataset.from_tensor_slices(list(te_traj))
    preds = model.predict(te_traj, batch_size=128)

    for label in unq_labels:
        fc_all = list()

        # METRICS
        fc_true_positive = 0
        fc_false_negative = 0
        fc_false_positive = 0

        is_label = labels == label
        is_preds = preds[is_label]
        is_grf = te_grf[is_label]

        # Loop over every Validation-Data Trial
        for j in range(0, is_preds.shape[0]):
            pred = is_preds[j]

            # Loop over each Output-Channel starting with L_IC -> 1; L_FO -> 2; R_IC -> 3; R_FO -> 4;
            for k in range(1, pred.shape[1]):
                if k == 2:
                    grf_events = np.array(np.where(is_grf[j] == (k+1)))
                else:
                    grf_events = np.array(np.where(is_grf[j] == 3)) # Initial Contact in TE_GRF_EVENTS is still 3!



                # Find all peaks, where the probability is higher then min_peak_threshold (0.3)
                [loc, height] = find_peaks(pred[:, k], height=min_peak_threshold, distance=50)

                for l in range(0, grf_events.shape[1]):

                    # When no peak is detected/predicted -> False Negative
                    if not loc.any():
                        print("False Negative - No Event Predicted")

                        if k == 1 or k == 2:
                            fc_false_negative += 1

                    # Peaks are detected/predicted
                    else:
                        distances = abs(loc - grf_events[0][l])

                        if sum(distances <= ground_truth_threshold) <= 1:
                            closest_index = np.argmin(abs(loc - grf_events[0][l]))
                            distance = loc[closest_index] - grf_events[0][l]
                        else:
                            idx = [i for i in range(len(distances)) if distances[i] <= ground_truth_threshold]
                            idx = idx[np.argmax(height['peak_heights'][idx])]
                            distance = loc[idx] - grf_events[0][l]

                        # When Ground Truth and Predicted Event Distance < +/- 5 -> True Positive
                        if abs(distance) < ground_truth_threshold:
                            if k == 1 or k == 2:
                                fc_all.append(distance)
                                fc_true_positive += 1
                        # Predicted Event is more then 5 Frames off from the Ground Truth
                        else:

                            # When Predicted Event is in a +/- 50 Frame Window --> False Positive
                            if abs(distance) <= 50:
                                if k == 1 or k == 2:
                                    fc_false_positive += 1
                                    fc_all.append(distance)
                                    print(f"Label:{label}, number: {j}, side: {k}, length: {distance}")

                            # When Predicted Event is above the +/- 50 Frame Window --> False Negative
                            else:
                                if k == 1 or k == 2:
                                    fc_false_negative += 1

        if np.array(fc_all).shape[0] > 0:                          
            fc_mae.append(mean_absolute_error(np.zeros(np.array(fc_all).shape), fc_all))
        else:
            fc_mae = 999 # Something is wrong

        fc_list_all.append(fc_all)
        fc_tp.append(fc_true_positive)
        fc_fp.append(fc_false_positive)
        fc_fn.append(fc_false_negative)

        fc_classification = pd.DataFrame([[fc_tp, fc_fp, fc_fn]])
    return fc_mae, fc_list_all, fc_classification

##########################################################################################################################################

##########################################################################################################################################

# Function: ' eval_fo_te_data:

    # Purpose: 
        # The function evaluates the performance of the input model on the test dataset. 
        #  It calculates various metricss for the unique labels in the labels array (for different pathologies separately).

    # Inputs:
        # model: a TensorFlow Keras model
        # te_traj: a numpy array of shape (num_samples, num_frames, num_features) containing the test trajectory data
        # te_grf: a numpy array of shape (num_samples, num_frames) containing the GRF data for the test trials
        # labels: a numpy array of shape (n_samples,) containing integer labels for each trial in te_traj to identify 
        # results for different pathologies
        
    # The function sets two threshold parameters:
        # min_peak_threshold: a float representing the minimum peak threshold for detecting GRF events
        # ground_truth_threshold: an integer representing the maximum allowed distance between predicted and ground truth event frames
    
    # Returns:
        # the mean absolute error over all true positive and false positive events
        # the list of all events as a distance to the ground truth events (0 = zero frames off, 1 = one frame off, and so on)
        # the classifications (true positive, false positive and false negative)
        # In a list segmented for each unique label (= each pathology)
 
    # Functionality:
        # The function then loops over each trial in the test dataset, and for each trial, loops over each output channel 
        # of the prediction. For each output channel, the function finds all peaks in the corresponding prediction 
        # with a height greater than 'min_peak_threshold'. It then compares the detected peaks to the ground truth event
        # frames too the corresponding GRF events.

        # If no peak is detected, the function considers this a false negative event. 
        # If at least one peak is detected, the function calculates the distance between each detected peak and the nearest ground
        # truth event frame. If the distance is less than or equal to 'ground_truth_threshold', the event is considered a true positive.
        # If the distance is greater than ground_truth_threshold but less than or equal to 50 frames, the event is 
        # considered a false positive. If the distance is greater than 50 frames, the event is considered a false negative.
        
def eval_fo_te_data(model, te_traj, te_grf, labels):
    fo_list_all = list()
    fo_mae = list()
    fo_tp = list()
    fo_fp = list()
    fo_fn = list()
    
    unq_labels = np.unique(labels)
    # THRESHOLD PARAMETERS
    min_peak_threshold = 0.01
    ground_truth_threshold = 5

    # Predict Test-Data
    te_traj = [traj.reshape(1, te_traj.shape[1], te_traj.shape[2]) for traj in te_traj]
    te_traj = tf.data.Dataset.from_tensor_slices(list(te_traj))
    preds = model.predict(te_traj, batch_size=128)

    for label in unq_labels:
        fo_all = list()

        # METRICS
        fo_true_positive = 0
        fo_false_negative = 0
        fo_false_positive = 0

        is_label = labels == label
        is_preds = preds[is_label]
        is_grf = te_grf[is_label]

        # Loop over every Validation-Data Trial
        for j in range(0, is_preds.shape[0]):
            pred = is_preds[j]

            # Loop over each Output-Channel starting with L_IC -> 1; L_FO -> 2; R_IC -> 3; R_FO -> 4;
            for k in range(1, pred.shape[1]):
                if k == 1:
                    grf_events = np.array(np.where(is_grf[j] == (k+1)))
                elif k == 2:
                    grf_events = np.array(np.where(is_grf[j] == (k+2))) # Foo Off in TE_GRF_EVENTS is still 4!



                # Find all peaks, where the probability is higher then min_peak_threshold (0.3)
                [loc, height] = find_peaks(pred[:, k], height=min_peak_threshold, distance=50)

                for l in range(0, grf_events.shape[1]):

                    # When no peak is detected/predicted -> False Negative
                    if not loc.any():
                        print("False Negative - No Event Predicted")

                        if k == 1 or k == 2:
                            fo_false_negative += 1

                    # Peaks are detected/predicted
                    else:
                        distances = abs(loc - grf_events[0][l])

                        if sum(distances <= ground_truth_threshold) <= 1:
                            closest_index = np.argmin(abs(loc - grf_events[0][l]))
                            distance = loc[closest_index] - grf_events[0][l]
                        else:
                            idx = [i for i in range(len(distances)) if distances[i] <= ground_truth_threshold]
                            idx = idx[np.argmax(height['peak_heights'][idx])]
                            distance = loc[idx] - grf_events[0][l]

                        # When Ground Truth and Predicted Event Distance < +/- 5 -> True Positive
                        if abs(distance) < ground_truth_threshold:
                            if k == 1 or k == 2:
                                fo_all.append(distance)
                                fo_true_positive += 1
                        # Predicted Event is more then 5 Frames off from the Ground Truth
                        else:

                            # When Predicted Event is in a +/- 50 Frame Window --> False Positive
                            if abs(distance) <= 50:
                                if k == 1 or k == 2:
                                    fo_false_positive += 1
                                    fo_all.append(distance)
                                    print(f"Label:{label}, number: {j}, side: {k}, length: {distance}")

                            # When Predicted Event is above the +/- 50 Frame Window --> False Negative
                            else:
                                if k == 1 or k == 2:
                                    fo_false_negative += 1

        if np.array(fo_all).shape[0] > 0:                          
            fc_mae.append(mean_absolute_error(np.zeros(np.array(fo_all).shape), fo_all))
        else:
            fo_mae = 999 # Something is wrong

        fo_list_all.append(fc_all)
        fo_tp.append(fc_true_positive)
        fo_fp.append(fc_false_positive)
        fo_fn.append(fc_false_negative)

        fo_classification = pd.DataFrame([[fo_tp, fo_fp, fo_fn]])
    return fo_mae, fo_list_all, fo_classification