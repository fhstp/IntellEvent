import numpy as np
from scipy.signal import find_peaks
import http.client
import json
import pandas as pd
import os
import sys

# Define a threshold [0-1] when events should be detected based on the prediction probability
# Increase min_peak_threshold if ghost events appear
# "Real" IC/FO events should be > 0.5.
min_peak_threshold = 0.2
base_frequency = 150


def reshape_data(traj):
    """
    Reshape a 2D numpy array into the required input format for the neural network.

    Parameters:
        traj (numpy.ndarray): A 2D array of shape (num_features, num_frames).

    Returns:
        numpy.ndarray: Reshaped array of shape (num_samples, num_frames, num_features).
    """
    rs_traj = np.transpose(np.array(traj).reshape(1, traj.shape[0], traj.shape[1]), (0, 2, 1))
    return rs_traj


def load_config(config_file='config.json'):
    """
    Load configuration from a JSON file.

    Parameters:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration with 'host' and 'port'.
    """
    if getattr(sys, 'frozen', False):  # If running as a PyInstaller bundle
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(base_path, config_file)

    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file '{config_path}' not found. Using default settings.")
        return {"host": "127.0.0.1", "port": 5000}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{config_path}': {e}")
        return {"host": "127.0.0.1", "port": 5000}


def send_prediction_request(endpoint, rs_velo):
    """
    Send a prediction request to the HTTP server.

    Parameters:
        endpoint (str): The endpoint to connect to (e.g., "/predict_ic" or "/predict_fo").
        rs_velo (numpy.ndarray): Reshaped velocity data to send to the server.

    Returns:
        numpy.ndarray: Prediction probabilities returned by the server.
    """
    # Load the host and port from the configuration file
    config = load_config()
    host = config.get('host', '127.0.0.1')  # Default to localhost if not specified
    port = config.get('port', 5000)  # Default to port 5000 if not specified

    conn = http.client.HTTPConnection(host, port)
    payload = json.dumps({'traj': rs_velo})
    headers = {'Content-Type': 'application/json'}

    conn.request("POST", endpoint, body=payload, headers=headers)
    response = conn.getresponse()

    if response.status != 200:
        raise Exception(f"Server error: {response.status}, {response.read().decode()}")

    predictions = json.loads(response.read())
    conn.close()
    return np.array(predictions)


def ic_pred(rs_velo, x_traj, subject_name, start_frame, vicon, cam_frequency):
    """
    Send data to the server to predict IC events and set them in Vicon Nexus.
    """
    ic_preds = send_prediction_request("/predict_ic", rs_velo)
    set_ic_events(ic_preds[0], x_traj[0, :], x_traj[3, :], subject_name, start_frame, vicon, cam_frequency)
    return True


def fo_pred(rs_velo, x_traj, subject_name, start_frame, vicon, cam_frequency):
    """
    Send data to the server to predict FO events and set them in Vicon Nexus.
    """
    fo_preds = send_prediction_request("/predict_fo", rs_velo)
    set_fo_events(fo_preds[0], x_traj[0, :], x_traj[3, :], subject_name, start_frame, vicon, cam_frequency)
    return True


def set_ic_events(ic_preds, l_heel, r_heel, subject_name, start_frame, vicon, cam_frequency):
    """
    Set IC events in Vicon Nexus using the predictions.
    """
    loc, _ = find_peaks(ic_preds[:, 1], height=min_peak_threshold, distance=25)
    loc = np.ceil((loc / base_frequency) * cam_frequency).astype(int)
    for ic in loc:
        if l_heel[ic] < r_heel[ic]:
            vicon.CreateAnEvent(subject_name, "Left", "Foot Strike", int(ic + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name, "Right", "Foot Strike", int(ic + start_frame), 0.0)


def set_fo_events(fo_preds, l_heel, r_heel, subject_name, start_frame, vicon, cam_frequency):
    """
    Set FO events in Vicon Nexus using the predictions.
    """
    loc, _ = find_peaks(fo_preds[:, 1], height=min_peak_threshold, distance=25)
    loc = np.ceil((loc / base_frequency) * cam_frequency).astype(int)
    for fo in loc:
        if l_heel[fo] > r_heel[fo]:
            vicon.CreateAnEvent(subject_name, "Left", "Foot Off", int(fo + start_frame), 0.0)
        else:
            vicon.CreateAnEvent(subject_name, "Right", "Foot Off", int(fo + start_frame), 0.0)


def get_trial_infos(c3d_trial):
    """
    Extract trial information from the C3D trial data.
    """
    cam_frequency = c3d_trial['parameters']['POINT']['RATE']['value'][0]
    return cam_frequency


def resample_data(traj, sample_frequ, frequ_to_sample):
    """
    Resample the data to the desired frequency.
    """
    period = '{}N'.format(int(1e9 / sample_frequ))
    index = pd.date_range(0, periods=len(traj[0, :]), freq=period)
    resampled_data = [pd.DataFrame(val, index=index).resample('{}N'.format(int(1e9 / frequ_to_sample))).mean() for val
                      in traj]
    resampled_data = [np.array(traj.interpolate(method='linear')) for traj in resampled_data]
    resampled_data = np.concatenate(resampled_data, axis=1)
    return resampled_data