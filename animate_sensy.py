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

## INFORMATION
# This script is to visualise 3D HPE (Human Pose Estimation) in 3D space. It can handle input as XYZ 3D positional coordinates from any output system (e.g. mocap, IMU) as long as they are saved as
# they are in a matrix format of frames x keypoints (<keypoint_1>_x, <keypoint_1>_y, <keypoint_1>_z, <keypoint_2>_x, ... <keypoint_n>_z) 

## SETUP
plt.ioff()
plt.style.use('dark_background')

# False if from IMU data
flag_seperateXYZ    = True # False for XSens (position data split to x, y, z files)
flag_makeGIF        = True
flag_midShldrPevlis = False
flag_remOffset      = False
# Select at what marker to take offset. This will stop "floating"
offset_marker       = 'rfoo_x' #lank_smpl_x'#'LAnkJnt_positionX'#'lank_smpl_x' #'ankleRightX'
flag_filter         = False
flag_rotate         = False # to Allign Theia and METRABS

file_identifier = '.csv'

scale_factor = 1000

# Filtering parameters         
f_order = 4
f_cutoff = 1
f_sampling = 30
f_nyquist = f_cutoff/(f_sampling/2)
b, a = signal.butter(f_order, f_nyquist, btype='lowpass', analog = False)

# Where to read data from
data_path_in = '../OrthoCAP/In/' #'/Users/orthocap_01/Library/CloudStorage/Box-Box/Data For Analysis/HPC Iphone Recordings/1027 LYEL (9.27.24 HPC) [Control Participant]/metrabs/' #'../Study_XSens_Ames/In/Box_data/' #'../Study_Erin_spine/Out/metrabs/SP1002/' #'../Study_Validation/Out/metrabs/' # #'./Out/Data/HPC_tests/'

# Check if ./Figures/ path exists if not make folder
if not os.path.exists(data_path_in + 'Figures/'):
    os.mkdir(data_path_in + 'Figures/')

# List files in directory, loop through them and check for .csv
csv_files = os.listdir(data_path_in)

for i_csv_file in csv_files:
    if i_csv_file.endswith(file_identifier): #3DTracked

        # Load in tracked joint data from 3D pose estimation
        if flag_seperateXYZ == True:

            # Get base trial name
            trial_name = data_path_in + i_csv_file

            data_xyz = pd.read_csv(trial_name)

            # Check if Column exists and remove it
            data_headers = data_xyz.columns
            if data_headers.__contains__('Unnamed: 0'):
                data_xyz = data_xyz.drop(columns='Unnamed: 0')
            elif data_headers.__contains__('timestamp'):
                data_xyz = data_xyz.drop(columns='timestamp')
                data_headers = data_headers.drop('timestamp')
            
            
        
            

            # TO np.array
            pose_xyz = np.array(data_xyz, dtype='float')
            n_frames, n_markers = pose_xyz.shape
                     
            # ## CHECK ##
            # yf = rfft((pose_y[:,10] - np.average(pose_y[:,10]))/1e6)
            # xf = rfftfreq(n_frames, 1/30)
            # plt.plot(xf, np.abs(yf))

            # Filter Keypoints
            if flag_filter == True:
                pose_xyz = signal.filtfilt(b, a, pose_xyz, axis=0)

            # yf = rfft((pose_y[:,10] - np.average(pose_y[:,10]))/1e6)
            # xf = rfftfreq(n_frames, 1/30)
            # plt.plot(xf, np.abs(yf))

            # Remove offset
            if flag_remOffset == True:
                # Create copy of matrix object otherwise it will follow what is happening to the object
                idx_offset_marker = data_xyz.columns.get_loc(offset_marker)
                pose_off = np.copy(pose_xyz[:,idx_offset_marker:idx_offset_marker+3])
                # pose_x_off = np.average(pose_x, axis = 1)
                pose_off.shape = [n_frames,3]
                for i_col in range(int(n_markers/3)):
                    pose_xyz[:,i_col*3:i_col*3+3] = pose_xyz[:,i_col*3:i_col*3+3] - pose_off
                
            # Apply manual rotation (allign Theia and inference)
            if flag_rotate == True:
                r_mat = R.from_rotvec(np.pi/5 * np.array([0, 0, 1])) 
                for i_marker in range(0,30):
                    pose_xyz[:,3*i_marker:3*i_marker+3] = r_mat.apply(pose_xyz[:,3*i_marker:3*i_marker+3])   

            # Save out processed data
            out_xyz = pd.DataFrame(pose_xyz)
            # out_xyz.to_csv((data_path_in + i_csv_file[:-4] + '_pro.csv'), header = data_headers)

            
            # Split
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

            if np.isnan(z_min):
                z_min = -0.1

            if np.isnan(z_max):
                z_max=2.5

        # Load in data from IMU estimations (Movella)
        else:

            # Get base trial name
            trial_name = data_path_in + i_csv_file[:-8] #'IMU_Segment_pos_xyz'
            # Get position .csv
            data_xyz = pd.read_csv(trial_name + '_pos.csv')
            # Remove "Frame" column
            data_xyz = data_xyz.drop(columns='Frame')
 
            # Remove "Shoulder" (blade) columns
            data_xyz = data_xyz.drop(columns='Right Shoulder x')
            data_xyz = data_xyz.drop(columns='Right Shoulder y')
            data_xyz = data_xyz.drop(columns='Right Shoulder z')
            data_xyz = data_xyz.drop(columns='Left Shoulder x')
            data_xyz = data_xyz.drop(columns='Left Shoulder y')
            data_xyz = data_xyz.drop(columns='Left Shoulder z')

            # Get indeces of Pelvis and L/R Shoulders (Upper Arm segment) for position
            idx_pelvis_p = int(data_xyz.columns.get_loc('Pelvis x'))
            idx_shld_r_p = int(data_xyz.columns.get_loc('Right Upper Arm x'))
            idx_shld_l_p = int(data_xyz.columns.get_loc('Left Upper Arm x')) 

            # TO np.array and into mm
            pose_xyz = np.array(data_xyz, dtype='float')*1000

            # split data into X, Y, Z
            pose_x = pose_xyz[:,0::3]
            x_min = np.min(pose_x)
            x_max = np.max(pose_x)
            
            pose_y = pose_xyz[:,1::3]
            y_min = np.min(pose_y)
            y_max = np.max(pose_y)

            pose_z = pose_xyz[:,2::3]
            z_min = np.min(pose_z)
            z_max = np.max(pose_z)



            # Get quaternion .csv     
            # Load in tracked joint data from 3D orientation and pass to array (quaternions)    
            data_q0123 = pd.read_csv(trial_name + '_qua.csv')  
            data_q0123 = data_q0123.drop(columns='Frame')      

            # Get indeces of Pelvis or quaterinion
            idx_pelvis_q = int(data_q0123.columns.get_loc('Pelvis q0'))

            # To np.array quaternion
            ori_quat = np.array(data_q0123, dtype='float')

            # split data into q0, q1, q2, q3
            # MVN uses Scalar First for quaternion and SciPy uses Scalar Last for quaternion
            ori_q0 = ori_quat[:,0::4]                       
            ori_q1 = ori_quat[:,1::4] 
            ori_q2 = ori_quat[:,2::4]
            ori_q3 = ori_quat[:,3::4]

            # Check both position and quat have the same frames        
            n_frames_pos, n_cols_pos = np.shape(data_xyz)
            n_frames_quat, n_cols_quat = np.shape(data_q0123)

            if n_frames_pos == n_frames_quat:
                n_frames = n_frames_pos
            else:
                n_frames = np.min([n_frames_pos, n_frames_quat])

        # Calculate delta(mid shoulder, pelvis)
        if flag_midShldrPevlis == True:
            # ! should find better way of indexing data with header info from .mvn
            pelvis = pose_xyz[:,idx_pelvis_p:idx_pelvis_p + 3]
            shoulder_r = pose_xyz[:,idx_shld_r_p:idx_shld_r_p + 3]
            shoulder_l = pose_xyz[:,idx_shld_l_p:idx_shld_l_p + 3]
            midShoulder = np.mean(np.array([shoulder_r, shoulder_l]), axis=0)
            # Calculate vector between pelvis and mid shoulders (in global frame)
            d_mShd2Pel = midShoulder - pelvis

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

                if flag_midShldrPevlis == True:
                    ax.scatter(midShoulder[i,0], midShoulder[i,1], midShoulder[i,2], c = 'green', s = 14, marker = 'o' )

                    # Orientation of pelvis in global from IMU quaternions
                    rm_pelvis = R.from_quat([ori_q1[i, idx_pelvis_q], ori_q2[i, idx_pelvis_q], ori_q3[i, idx_pelvis_q], ori_q0[i, idx_pelvis_q]])
                    ang_eul[i,:] = rm_pelvis.as_euler('xyz', degrees=True)
                    
                    # Get as rotation matrix to calculate vectors
                    rm_pelvis_mat = rm_pelvis.as_matrix()
                    # Get pelvis vectors for coordinate system plotting
                    pelvis_pos = [pose_x[i,idx_pelvis_q], pose_y[i,idx_pelvis_q], pose_z[i,idx_pelvis_q]]
                    pelvis_rf_pnts = np.transpose(rm_pelvis_mat)*100 + pelvis_pos

                    # Plot pelvis coordinate system
                    ax.plot([pelvis_pos[0],pelvis_rf_pnts[0,0]], [pelvis_pos[1],pelvis_rf_pnts[0,1]], [pelvis_pos[2],pelvis_rf_pnts[0,2]],color = 'red')
                    ax.plot([pelvis_pos[0],pelvis_rf_pnts[1,0]], [pelvis_pos[1],pelvis_rf_pnts[1,1]], [pelvis_pos[2],pelvis_rf_pnts[1,2]],color = 'green')
                    ax.plot([pelvis_pos[0],pelvis_rf_pnts[2,0]], [pelvis_pos[1],pelvis_rf_pnts[2,1]], [pelvis_pos[2],pelvis_rf_pnts[2,2]],color = 'blue')
                
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
                # if flag_seperateXYZ:
                #     cons = np.loadtxt(data_path_in + i_csv_file[:-14] + '_edges.txt', dtype=int)
                #     n_cons = cons.__len__()

                #     for i_con in range(0, n_cons):
                #         ax.plot([x[cons[i_con][0]], x[cons[i_con][1]]], [y[cons[i_con][0]],y[cons[i_con][1]]], [z[cons[i_con][0]], z[cons[i_con][1]]],color = 'green')
                # # Hard code for XSens connections
                # else:
                #     cons = [
                #             [0,1],
                #             [1,2],
                #             [2,3],
                #             [3,4],
                #             [4,5],
                #             [5,6], #neck - head
                #             [4,7],
                #             [7,8],
                #             [8,9],
                #             [4,10],
                #             [10,11],
                #             [11,12],
                #             [0,13],
                #             [13,14],
                #             [14,15],
                #             [15,16],
                #             [0,17],
                #             [17,18],
                #             [18,19], 
                #             [19,20]                                                   
                #             ]
                #     n_cons = cons.__len__()

                #     for i_con in range(0, n_cons):
                #         ax.plot([x[cons[i_con][0]], x[cons[i_con][1]]], [y[cons[i_con][0]],y[cons[i_con][1]]], [z[cons[i_con][0]], z[cons[i_con][1]]],color = 'green')
            
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