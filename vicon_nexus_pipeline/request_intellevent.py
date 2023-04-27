# Note: tensorflow, numpy and pandas MUST have the described versions
# otherwise the mode cannot be opened and the code won´t work.

from viconnexusapi import ViconNexus
# cd C:\Program Files\Vicon\Nexus2.14\SDK\Win64\Python
#pip install ./viconnexusapi
from sklearn import preprocessing
import numpy as np #conda install numpy==1.18.5
from ezc3d import c3d #conda install -c conda-forge ezc3d
from utils import reshape_data, fc_pred, fo_pred
import time
import threading
import pandas as pd #conda install pandas==1.2.0

# marker names which are used for the algorithm
# adapt names to how they are stored in your .c3d file
marker_list = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]

if __name__=='__main__':
    # timer = time.time()

    # Get Vicon Nexus specific information from the viconnexus API
    vicon = ViconNexus.ViconNexus()
    path, file_name = vicon.GetTrialName()
    # opens the current .c3d file which is open in Vicon Nexus
    c3d_trial = c3d(path + file_name + '.c3d', extract_forceplat_data=True)
    subject_name = vicon.GetSubjectNames()
    trial_information = c3d_trial['parameters']['TRIAL']
    start_frame = trial_information['ACTUAL_START_FIELD']['value'][0]

    # Get all marker label names from the .c3d file
    labels = c3d_trial['parameters']['POINT']['LABELS']['value']
    # Get the corresponding index for each marker name in 'marker_list'
    marker_index = [labels.index(label) for label in marker_list]

    # Get all trajectories from the .c3d file
    trajectory_list = c3d_trial['data']['points']

    # Get the x, y, and z trajectories corresponding to the 'marker_index' list
    # Note: x-axis = direction of movement, y-axis = mediolateral movement, z-axis = vertical movement
    x_traj = trajectory_list[0, marker_index, :]
    y_traj = trajectory_list[1, marker_index, :]
    z_traj = trajectory_list[2, marker_index, :]

    # The current best model uses the x and z axis velocity for the IC model
    # and the x, y, and z axis velocity for the FO model
    fc_traj = np.concatenate([x_traj, z_traj])
    fo_traj = np.concatenate([x_traj, y_traj, z_traj])

    # x and y-coordinates need to be standardized depending on the starting direction,
    # z coordinates are always the same
    if any(fc_traj[0, 0:10] < 0):
        fc_traj[0:6, :] = (fc_traj[0:6, :] - np.mean(fc_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)
    else:
        fc_traj[0:6, :] = fc_traj[0:6, :] - np.mean(fc_traj[0:6, :], axis=1).reshape(6,1)
        fo_traj[0:12, :] = fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)

    fc_traj[6:12, :] = fc_traj[6:12, :] - np.mean(fc_traj[6:12, :], axis=1).reshape(6,1)
    fo_traj[12:18, :] = fo_traj[12:18, :] - np.mean(fo_traj[12:18, :], axis=1).reshape(6,1)

    # calculate the first derivative (= velocity) of the trajectories
    fc_velo = np.gradient(fc_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
    fc_velo = preprocessing.minmax_scale(fc_velo, feature_range=(0.1, 1.1), axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)

    # both 'fc_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
    # for the prediction we need the shape of (num_samples, num_frames, num_features)
    # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
    # check with rs_fc_velo.shape
    rs_fc_velo = reshape_data(fc_velo)
    rs_fo_velo = reshape_data(fo_velo)

    # Down / up sampling?
    #grf_frequency, cam_frequency = get_trial_infos(c3d_trial)
    # %% Down Sampling NOT OSS
    #period = '{}N'.format(int(1e9 / 120))
    #index = pd.date_range(0, periods=len(rs_velo[0, :]), freq=period)
    #new_data = [pd.DataFrame(val, index=index).resample('{}N'.format(int(1e9 / 150))).mean() for val in rs_velo]
    #rs_traj_velo = reshape_data(new_data[0]).tolist()
    #new_data = new_data[0].interpolate(method='linear')
    #rs_velo = np.transpose(np.array(new_data).reshape(1, new_data.shape[0], new_data.shape[1]), (0, 1, 2)).tolist()

    # Multithreading to run both predictions at the same time
    # speeds up processing
    t1 = threading.Thread(target=fc_pred, args=(rs_fc_velo.tolist(), fc_traj[0:6], subject_name, start_frame, vicon))
    t2 = threading.Thread(target=fo_pred, args=(rs_fo_velo.tolist(), fc_traj[0:6], subject_name, start_frame, vicon))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    #print(time.time() - timer)
    vicon.Disconnect()