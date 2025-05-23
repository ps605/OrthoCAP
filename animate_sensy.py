import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import linalg as la
import matplotlib.animation as animation
from scipy.fft import fft, fftfreq, rfft, rfftfreq
import csv
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.cm as cm
import os
from spectralAnalysis import spectralAnalysis

## INFORMATION
# This script is to visualise 3D HPE (Human Pose Estimation) in 3D space. It can handle input as XYZ 3D positional coordinates from any output system (e.g. mocap, IMU) as long as they are saved as
# they are in a matrix format of frames x keypoints (<keypoint_1>_x, <keypoint_1>_y, <keypoint_1>_z, <keypoint_2>_x, ... <keypoint_n>_z) 

## SETUP
plt.ioff()
plt.style.use('dark_background')

# False if from IMU data
flag_seperateXYZ    = True 
flag_makeGIF        = True
flag_filter         = False
flag_rotate         = False 
flag_remOffset      = False
# Select at what marker to take offset. This will stop "floating"
offset_marker       = 'rfoo_x' #lank_smpl_x'#'LAnkJnt_positionX'#'lank_smpl_x' #'ankleRightX'

file_identifier = '.csv'

scale_factor = 1

# Filtering parameters         
f_order = 4
f_cutoff = 1
f_sampling = 30
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(f_order, f_nyquist, btype='lowpass', analog = False)

# Where to read data from
data_path_in = '../In/'
# Where to write data to
data_path_out = '../Out/'

# Check if ./Figures/ path exists if not make folder
if not os.path.exists(data_path_in + 'Figures/'):
    os.mkdir(data_path_in + 'Figures/')

# Check if ./Out/Processed path exists if not make folder
if not os.path.exists(data_path_out + 'Processed/'):
    os.mkdir(data_path_out + 'Processed/')

# List files in directory, loop through them and check for .csv
csv_files = os.listdir(data_path_in)

for i_csv_file in csv_files:
    if i_csv_file.endswith(file_identifier): 

        # Load in tracked joint data from 3D pose estimation
        if flag_seperateXYZ == True:

            # Get base trial name
            trial_name = data_path_in + i_csv_file

            data_xyz = pd.read_csv(trial_name)

            # Check if Column exists and remove it
            data_headers = data_xyz.columns

            if data_headers.__contains__('timestamp'):
                data_xyz = data_xyz.drop(columns='timestamp')
                data_headers = data_headers.drop('timestamp')
            
            # Get Data shape
            pose_xyz = np.array(data_xyz, dtype='float')
            n_frames, n_markers = pose_xyz.shape
                  
            # Filter Keypoints
            if flag_filter == True:
                pose_xyz = signal.filtfilt(b, a, pose_xyz, axis=0)

            # Remove offset
            if flag_remOffset == True:
                # Create copy of matrix object otherwise it will follow what is happening to the object
                idx_offset_marker = data_xyz.columns.get_loc(offset_marker)
                pose_off = np.copy(pose_xyz[:,idx_offset_marker:idx_offset_marker+3])
                
                pose_off.shape = [n_frames,3]
                for i_col in range(int(n_markers/3)):
                    pose_xyz[:,i_col*3:i_col*3+3] = pose_xyz[:,i_col*3:i_col*3+3] - pose_off
                
            # Apply manual rotation (allign Theia and inference)
            if flag_rotate == True:
                # Can define the r_mat directly from rotational matrix also 
                r_mat = R.from_rotvec(np.pi/5 * np.array([0, 0, 1])) 
                for i_marker in range(0,n_markers):
                    pose_xyz[:,3*i_marker:3*i_marker+3] = r_mat.apply(pose_xyz[:,3*i_marker:3*i_marker+3])   

            # Save out processed data
            out_xyz = pd.DataFrame(pose_xyz)
            out_xyz.to_csv((data_path_out + i_csv_file[:-4] + '_pro.csv'), header = data_headers)
            
            # Split Data to X, Y, Z
            pose_xyz=pose_xyz * scale_factor
            pose_x = pose_xyz[:,0::3]         
            pose_y = pose_xyz[:,1::3]       
            pose_z = pose_xyz[:,2::3]

            x_min = np.min(pose_x)
            x_max = np.max(pose_x)
            y_min = np.min(pose_y)
            y_max = np.max(pose_y)
            z_min = np.min(pose_z)
            z_max = np.max(pose_z)

            spectralAnalysis(pose_x[:,0], 1/30)

            if np.isnan(z_min):
                z_min = -0.1

            if np.isnan(z_max):
                z_max=2.5

        n_frames, n_cols = np.shape(pose_x)
        frames_v = range(n_frames)
        ang_eul = np.zeros((n_frames,3))


        # Generate .gif of motion
        if flag_makeGIF:
            data_frames, data_joints = pose_x.shape
            track_marker_idx = []

            # Generate empty arrays for ploting joint paths
            X = []
            Y = []
            Z = []

            # Select colour map
            colours = cm.prism(np.linspace(0, 1 , data_joints))

            # Plotting update function to iterate through frames
            def update(i):
                ax.cla()

                # Plot Global frame
                ax.plot([0,200], [0,0], [0,0],color = 'red')
                ax.plot([0,0], [0,200], [0,0],color = 'green')
                ax.plot([0,0], [0,0], [0,200],color = 'blue')
                ax.scatter(0, 0, 0, c = 'red', s = 15, marker = 'o')
                
                # Update position of segments points
                x = pose_x[i, :]
                y = pose_y[i, :]
                z = pose_z[i, :]

                ax.scatter(x, y, z, c = 'red', s = 15, marker = 'o')

                for i_joint in track_marker_idx:
                    X.append(x[i_joint])
                    Y.append(y[i_joint])
                    Z.append(0)
                    ax.plot(X, Y, Z, c = colours[i_joint])      

                ax.set_title('Frame Number:'  +  str(i))
                ax.set_xlim(x_min + 0.1*x_min, x_max + 0.1*x_max)
                ax.set_ylim(y_min + 0.1*y_min, y_max + 0.1*y_max)
                ax.set_zlim(z_min + 0.1*z_min, z_max + 0.1*z_max)
                ax.set_aspect('equal')
                ax.grid(False)   

                # Load and draw connections depending to the type of skeleton from METRABS
                
                cons = np.loadtxt('sency_edges.txt', dtype=int)
                n_cons = cons.__len__()

                for i_con in range(0, n_cons):
                    ax.plot([x[cons[i_con][0]], x[cons[i_con][1]]], [y[cons[i_con][0]],y[cons[i_con][1]]], [z[cons[i_con][0]], z[cons[i_con][1]]],color = 'green')
           
            fig = plt.figure(dpi=100)
            fig.set_figheight(9.6)
            fig.set_figwidth(12.8)
            ax = fig.add_subplot(projection='3d')
            
            # Create .git animation
            fig_name = i_csv_file[0:-4] #+ '_lp1_4order_offToe'
            
            ani = animation.FuncAnimation(fig = fig, func = update, frames = n_frames, interval = 1, repeat = False)
            writer = animation.PillowWriter(fps = f_sampling,
                                                metadata = 'None',  #dict(artist = 'Me')
                                                bitrate = 1000)   #1800
            ani.save(data_path_in + 'Figures/' + fig_name + '.gif', writer = writer )

            plt.close()
                
            print('Animation complete for:' + data_path_in + 'Figures/' + fig_name + '.gif')