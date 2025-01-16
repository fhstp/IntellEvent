from viconnexusapi import ViconNexus
from sklearn import preprocessing
import numpy as np #conda install numpy==1.18.5
from vicon_utils import ic_pred, fo_pred, reshape_data, resample_data
import threading
import sys
#import argparse

# marker names which are used for the algorithm
# adapt names to how they are called in Vicon
marker_list = ["LHEE", "LTOE", "LANK", "RHEE", "RTOE", "RANK"]
base_frequency = 150



if __name__=='__main__':
    x_traj, y_traj, z_traj = [], [], []
    #args_in = sys.argv

    # Get Vicon Nexus specific information from the viconnexus API
    vicon = ViconNexus.ViconNexus()
    vicon.ClearAllEvents()
    path, file_name = vicon.GetTrialName()
    subject_name = vicon.GetSubjectNames()[0]
    start_frame, end_frame = vicon.GetTrialRegionOfInterest()


    # Check for progression Axis
    marker = vicon.GetTrajectory(subject_name, "LHEE")
    prog_x = marker[0][start_frame - 1:end_frame - 1]
    prog_y = marker[1][start_frame - 1:end_frame - 1]

    # Get the corresponding index for each marker name in 'marker_list'

    try:
        for marker in marker_list:
            if vicon.HasTrajectory(subject_name, marker):
                x, y, z, _ = vicon.GetTrajectory(subject_name, marker)
                x_traj.append(x[start_frame - 1:end_frame - 1])
                y_traj.append(y[start_frame - 1:end_frame - 1])
                z_traj.append(z[start_frame - 1:end_frame - 1])
            else:
                xyz, _ = vicon.GetModelOutput(subject_name, marker)
                x_traj.append(xyz[0][start_frame - 1:end_frame - 1])
                y_traj.append(xyz[1][start_frame - 1:end_frame - 1])
                z_traj.append(xyz[2][start_frame - 1:end_frame - 1])
    except:
        print(f"No Marker with the name: {marker}")



    # The current best model uses the x and z axis velocity for the IC model
    # and the x, y, and z axis velocity for the FO model
    if np.mean(np.abs(prog_x)) > np.mean(np.abs(prog_y)):
        ic_traj = np.concatenate([x_traj, z_traj])
        fo_traj = np.concatenate([x_traj, y_traj, z_traj])
    else:
        ic_traj = np.concatenate([y_traj, z_traj])
        fo_traj = np.concatenate([y_traj, x_traj, z_traj])

    # x and y-coordinates need to be standardized depending on the starting direction,
    # z coordinates are always the same
    if any(ic_traj[0, 0:10] < 0) or any(ic_traj[3, 0:10] < 0):
        ic_traj[0:6, :] = (ic_traj[0:6, :] - np.mean(ic_traj[0:6, :], axis=1).reshape(6,1)) * (-1)
        fo_traj[0:12, :] = (fo_traj[0:12, :] - np.mean(fo_traj[0:12, :], axis=1).reshape(12, 1)) * (-1)

    # calculate the first derivative (= velocity) of the trajectories
    ic_velo = np.gradient(ic_traj, axis=1)
    fo_velo = np.gradient(fo_traj, axis=1)

    # standardize between 0.1 and 1.1 for the machine learning algorithm (zeros will be ignored!)
    ic_velo = preprocessing.minmax_scale(ic_velo, feature_range=(0.1, 1.1), axis=1)
    fo_velo = preprocessing.minmax_scale(fo_velo, feature_range=(0.1, 1.1), axis=1)


    #Down / up sampling?
    cam_frequency = vicon.GetFrameRate()
    if cam_frequency != base_frequency:
        rs_ic_velo = resample_data(ic_velo, cam_frequency, base_frequency).transpose()
        rs_fo_velo = resample_data(fo_velo, cam_frequency, base_frequency).transpose()
    else:
        rs_ic_velo = ic_velo
        rs_fo_velo = fo_velo

    # both 'ic_velo' and 'fo_velo' should be in the shape (num_features, num_frames) (e.g. (12, 500) or (18, 500))
    # for the prediction we need the shape of (num_samples, num_frames, num_features)
    # num_samples = 1, num_frames = length of trial (e.g. 500), num_features = velocity of trajectories (e.g. 12 or 18)
    # check with rs_ic_velo.shape
    rs_ic_velo = reshape_data(rs_ic_velo) #rs_ic_velo
    rs_fo_velo = reshape_data(rs_fo_velo) #rs_fo_velo


    # Multithreading to run both predictions at the same time
    # speeds up processing
    t1 = threading.Thread(target=ic_pred, args=(rs_ic_velo.tolist(), ic_traj[0:6], subject_name, start_frame, vicon, cam_frequency))
    t2 = threading.Thread(target=fo_pred, args=(rs_fo_velo.tolist(), ic_traj[0:6], subject_name, start_frame, vicon, cam_frequency))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    vicon.Disconnect()

