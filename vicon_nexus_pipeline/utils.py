import numpy as np
from scipy.signal import find_peaks
import requests

# Define a threshold [0-1] when events should be detected based on the prediction probability
# increase min_peak_threshold if ghost events appear
# "Real" IC/FO events should be > 0.5.
min_peak_threshold = 0.2


# Function: 'reshape_data':
    # Purpose:
        # This function reshapes a 2-dimensional numpy array (num_features, num_frames)
        # by transposing its dimensions into 3-dimensions (num_samples, num_frames, num_features)
    # Input:
        # 'traj': a numpy array of shape (num_features, num_frames) containing the velocities of the markerset
    # Output:
        # 'rs_traj': a numpy array of shape (num_samples, num_frames, num_features) containing the same
        # data but with the certain input format for the neural network
def reshape_data(traj):
    rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
    return rs_traj

# Function: 'fc_pred'
    # Purpose:
        # The function 'fc_pred' sends data to a server where a ML algorithm predicts IC events
        # and calls the function 'set_ic_events' for creating IC events in Vicon Nexus.

    # Input:
        # rs_velo: contains a list of a 3D array (num_samples, num_frames, num_features) representing the input for the neural network
        # x_traj: A 2D array of the x-axis (direction of movement) of the markers to identify which foot is in IC
        # subject_name: A string representing the name of the person.
        # start_frame: An integer representing the starting frame of the trial.
        # vicon: a viconnexus API object of the trial.

    # Output:
        # This function returns only Ture or False indicating whether the function executed successfully.

    # Description:
        # The 'fc_preds' function uses the 'rs_velo' as an input for a neural network model running on a Flask server
        # for a prediction of IC events using a HTTP POST request. The probailities of the neural network are returned as an array
        # 'fc_preds' and used as input for the 'set_ic_events' function to finally set IC events in the Vicon Nexus trial.

def fc_pred(rs_velo, x_traj, subject_name, start_frame, vicon):
    fc_preds = requests.post('http://127.0.0.1:5000/predict_fc', json={'traj': rs_velo})
    fc_preds = np.array(fc_preds.json())[0]
    set_ic_events(fc_preds, x_traj[0,:], x_traj[4,:], subject_name, start_frame, vicon)
    return True


# Function: 'fo_pred'
    # Purpose:
        # The function 'fo_pred' sends data to a server where a ML algorithm predicts FO events
        # and calls the function 'set_fo_events' for creating FO events in Vicon Nexus.

    # Input:
        # rs_velo: contains a list of a 3D array (num_samples, num_frames, num_features) representing the input for the neural network
        # x_traj: A 2D array of the x-axis (direction of movement) of the markers to identify which foot is in FO
        # subject_name: A string representing the name of the person.
        # start_frame: An integer representing the starting frame of the trial.
        # vicon: a viconnexus API object of the trial.

    # Output:
        # None - This function returns only Ture or False indicating whether the function executed successfully.

    # Description:
        # The 'fo_preds' function uses the 'rs_velo' as an input for a neural network model running on a Flask server
        # for a prediction of FO events using a HTTP POST request. The probailities of the neural network are returned as an array
        # 'fo_preds' and used as input for the 'set_fo_events' function to finally set FO events in the Vicon Nexus trial.
def fo_pred(rs_velo, x_traj, subject_name, start_frame, vicon):
    fo_preds = requests.post('http://127.0.0.1:5000/predict_fo', json={'traj': rs_velo})
    fo_preds = np.array(fo_preds.json())[0]
    set_fo_events(fo_preds, x_traj[0,:], x_traj[4,:], subject_name, start_frame, vicon)
    return True

# Function: 'set_ic_events'

    # Purpose:
        # This function sets the predicted IC events in the Vicon Nexus Software for the left and right foot.

    # Input:
        # fc_preds: contains the probabilities of the prediction from the neural network (num_frames, num_output_nodes).
        #           Output node [:, 0] contains the probabilities for a non-event. Output node [:, 1] contains the probabilities
        #           for the IC events.
        # l_heel: x-axis (direction of movement) trajectory of the left heel marker (num_samples). To identify the side of the predicted events.
        # r_heel: x-axis trajectory of the right heel marker (num_samples). To identify the side of the predicted events.
        # subject_name: A string representing the name of the person.
        # start_frame: An integer representing the starting frame of the trial.
        # vicon:  viconnexus API object of the trial.

    #Output:
        # None - This function returns only Ture or False indicating whether the function executed successfully.

    # Description:
    # This function first finds the peaks in the second column of 'fc_preds' (= probability for a IC event)
    # that have a height greater than or equal to the 'minimum peak threshold' and a distance of at least 25 frames between them,
    # using the find_peaks() function from the scipy library.
    # For each peak found, it determines which foot (left or right) is in front based on the x-coordinates of the
    # left and right heel markers at the frame of the event.
    # It then creates an event using the vicon.CreateAnEvent() function of the veiconnexus API object,
    # indicating the side of the foot (left/right) and the type of event (foot strike), at the corresponding frame number in the Vicon Nexus system.

def set_ic_events(fc_preds, l_heel, r_heel, subject_name, start_frame, vicon):
    [loc, height] = find_peaks(fc_preds[:, 1], height=min_peak_threshold, distance=25)
    for fc in loc:
        if l_heel[fc] < r_heel[fc]:
            vicon.CreateAnEvent(subject_name[0], "Left", "Foot Strike", int(fc + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name[0], "Right", "Foot Strike", int(fc + start_frame), 0.0)


# Function: 'set_fo_events'

    # Purpose:
    # This function sets the predicted FO events in the Vicon Nexus Software for the left and right foot.

    # Input:
        # fo_preds: contains the probabilities of the prediction from the neural network (num_frames, num_output_nodes).
        #           Output node [:, 0] contains the probabilities for a non-event. Output node [:, 1] contains the probabilities
        #           for the FO events.
        # l_heel: x-axis (direction of movement) trajectory of the left heel marker (num_samples). To identify the side of the predicted events.
        # r_heel: x-axis trajectory of the right heel marker (num_samples). To identify the side of the predicted events.
        # subject_name: A string representing the name of the person.
        # start_frame: An integer representing the starting frame of the trial.
        # vicon:  viconnexus API object of the trial.

    # Output:
        # None - This function returns only Ture or False indicating whether the function executed successfully.

    # Description:
        # This function first finds the peaks in the second column of 'fo_preds' (= probability for a FO event)
        # that have a height greater than or equal to the 'minimum peak threshold' and a distance of at least 25 frames between them,
        # using the find_peaks() function from the scipy library.
        # For each peak found, it determines which foot (left or right) is farther behind using the x-coordinates of the
        # left and right heel markers at the frame of the event.
        # It then creates an event using the vicon.CreateAnEvent() function of the veiconnexus API object,
        # indicating the side of the foot (left/right) and the type of event (foot off), at the corresponding frame number in the Vicon Nexus system.

def set_fo_events(fc_preds, l_heel, r_heel, subject_name, start_frame, vicon):
    [loc, height] = find_peaks(fc_preds[:, 1], height=min_peak_threshold, distance=25)
    for fo in loc:
        if l_heel[fo] > r_heel[fo]:
            vicon.CreateAnEvent(subject_name[0], "Left", "Foot Off", int(fo + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name[0], "Right", "Foot Off", int(fo + start_frame), 0.0)
