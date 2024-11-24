from pykalman import KalmanFilter
import numpy as np
import json
import os
import argparse
from evaluation_kalman import *


#def kalman_filter_3d_with_persistent_velocity(joints_3D, frame_rate=60, process_noise=8e-5, observation_noise=1e-4, frame_step=5):
def kalman_filter_3d_with_persistent_velocity(joints_3D, frame_rate=60, process_noise=1e-3, observation_noise=2e-3, frame_step=5):
    """
    Apply a Kalman filter to smooth the 3D joint data, including velocity estimation.
    The filter always uses the past velocity and retains it for future updates.
    
    Args:
    joints_3D: (N, 35, 3) array of joint positions over N frames in 3D space.
    frame_rate: Frame rate of the video (frames per second), default is 60.
    process_noise: Process noise covariance scalar. Higher values mean the model allows more state changes.
    observation_noise: Observation noise covariance scalar. Higher values mean more trust in the Kalman correction.
    frame_step: Number of frames to compute velocity over, default is 5.
    
    Returns:
    smoothed_poses: Smoothed 3D joint data.
    """
    
    N, num_joints, _ = joints_3D.shape

    if N < frame_step:
        return joints_3D
    
    # Compute velocity over `frame_step` frames
    delta_t = frame_step / frame_rate  # time difference between frames considered for velocity
    velocity = (joints_3D[frame_step:] - joints_3D[:-frame_step]) / delta_t  # velocity estimation
    
    # Pad velocity to match the shape (assume no velocity for the first few frames)
    velocity = np.vstack([np.zeros((frame_step, num_joints, 3)), velocity])
    
    # Flatten position and velocity to (N, 35 * 3 * 2) -> first 35*3 for position, next 35*3 for velocity
    flat_poses = joints_3D.reshape(N, -1)
    flat_velocity = velocity.reshape(N, -1)
    
    # Stack position and velocity for each frame
    state_vector = np.hstack([flat_poses, flat_velocity])
    
    # Define transition matrix with velocity update
    dim = state_vector.shape[1] // 2  # Number of positions (e.g., 105 for 35 joints, 3D)
    transition_matrix = np.block([
        [np.eye(dim), delta_t * np.eye(dim)],  # Position update with velocity
        [np.zeros((dim, dim)), np.eye(dim)]   # Velocity remains constant
    ])
    
    # Define observation matrix (only observe positions, not velocity)
    observation_matrix = np.hstack([np.eye(dim), np.zeros((dim, dim))])
    
    # Process noise covariance matrix (for both position and velocity)
    process_covariance = process_noise * np.eye(state_vector.shape[1])
    
    # Observation noise covariance matrix (only for positions)
    observation_covariance = observation_noise * np.eye(dim)
    
    # Initialize Kalman filter with the first frame as the initial mean
    kf = KalmanFilter(
        initial_state_mean=state_vector[0], 
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=process_covariance,
        observation_covariance=observation_covariance
    )
    
    filtered_state_covariance = process_covariance  # Start with process covariance

    # Initialize the smoothed_states list with the first state
    smoothed_states = [state_vector[0]]  # Start with the first frame

    # Apply Kalman filter with persistent velocity
    for i in range(1, N):
        # Use the previous frame's velocity without updating it
        state_vector[i, flat_poses.shape[1]:] = state_vector[i-1, flat_poses.shape[1]:]  # Retain previous velocity
        
        # Apply Kalman filter to smooth the data
        kf_state_mean, filtered_state_covariance = kf.filter_update(
            state_vector[i-1],
            filtered_state_covariance,  # Pass the previous state covariance matrix
            observation=flat_poses[i]
        )
        
        # Update the current state vector
        state_vector[i] = kf_state_mean
        smoothed_states.append(kf_state_mean)
    
    smoothed_states = np.array(smoothed_states)
    
    # Extract the smoothed positions (first 105 elements for 3D data)
    smoothed_positions = smoothed_states[:, :flat_poses.shape[1]]
    
    # Reshape the smoothed positions back to original format (N, 35, 3)
    smoothed_poses = smoothed_positions.reshape(N, num_joints, 3)
    
    return smoothed_poses




def adjust_using_previous_vector(current_ankle_pos, previous_ankle_pos, current_joint_pos, previous_joint_pos, threshold):
    # Calculer la distance actuelle entre la cheville et le joint
    distance = np.linalg.norm(current_joint_pos - current_ankle_pos)
    
    # Si la distance est trop grande ou trop petite, ajuster en utilisant le vecteur précédent
    if distance > threshold:
        # Calculer le vecteur de la frame précédente
        previous_vector = previous_joint_pos - previous_ankle_pos
        
        # Appliquer ce vecteur à la nouvelle position de la cheville
        adjusted_pos = current_ankle_pos + previous_vector
        return adjusted_pos
    return current_joint_pos  # Sinon, ne pas ajuster



def apply_kalman_filter_to_pose_data(frames, dict_metrics_3D, max_frame_gap=5):
    """
    Apply the Kalman filter to smooth pose data for each player, ensuring that 
    Kalman filtering is applied only to consecutive frames. If the gap between frames
    exceeds `max_frame_gap`, we apply the filter on separate pose segments.
    
    Args:
    frames: The collected pose data for each frame.
    max_frame_gap: Maximum allowable gap between frames before separating the data into segments.
    
    Returns:
    frames: Updated with smoothed pose data.
    """
    
    threshold_heel = 0.25  
    threshold_tip = 0.40   

    # Initialize variables to store consecutive segments
    current_segment = []
    all_segments = []


    # Loop through the pose data and collect consecutive frame segments
    for i, frame in enumerate(frames):
        if i == 0:
            current_segment.append(frame)
        else:
            prev_frame_number = frames[i - 1]['frame_number']
            curr_frame_number = frame['frame_number']
            
            # If the gap between frames is too large, save the current segment and start a new one
            if curr_frame_number - prev_frame_number > max_frame_gap:
                all_segments.append(current_segment)
                current_segment = []
            
            current_segment.append(frame)
    
    # Append the last segment
    if current_segment:
        all_segments.append(current_segment)
    

    frame_rate = 60

    incr = 0
    # Apply the Kalman filter to each consecutive segment
    for segment in all_segments:
        if len(segment) == 1:
            print("Skipping Kalman filter for a single-frame segment")
            frames[incr]['j3d'] = segment[0]['j3d']
            incr += 1
            continue

        # Extract the pose data for the current segment
        segment_poses = [frame['j3d'] for frame in segment]

        # Convert the pose data to a numpy array
        pose_data_np = np.array(segment_poses)
        before_kalman_pose = np.array([frame['j3d'] for frame in segment])

        for frame in range(1, pose_data_np.shape[0]):
            left_ankle = pose_data_np[frame, 6]  # Cheville gauche
            right_ankle = pose_data_np[frame, 3]  # Cheville droite
            left_heel = pose_data_np[frame, 31]  # Talon gauche
            right_heel = pose_data_np[frame, 22]  # Talon droit
            left_tip = pose_data_np[frame, 30]  # Pointe du pied gauche
            right_tip = pose_data_np[frame, 21]  # Pointe du pied droite
            
            # Frame précédente
            previous_left_ankle = pose_data_np[frame - 1, 6]
            previous_right_ankle = pose_data_np[frame - 1, 3]
            previous_left_heel = pose_data_np[frame - 1, 31]
            previous_right_heel = pose_data_np[frame - 1, 22]
            previous_left_tip = pose_data_np[frame - 1, 30]
            previous_right_tip = pose_data_np[frame - 1, 21]

            # Ajuster talon gauche et droit
            pose_data_np[frame, 31] = adjust_using_previous_vector(left_ankle, previous_left_ankle, left_heel, previous_left_heel, threshold_heel)
            pose_data_np[frame, 22] = adjust_using_previous_vector(right_ankle, previous_right_ankle, right_heel, previous_right_heel, threshold_heel)
            
            # Ajuster pointe du pied gauche et droit
            pose_data_np[frame, 30] = adjust_using_previous_vector(left_ankle, previous_left_ankle, left_tip, previous_left_tip, threshold_tip)
            pose_data_np[frame, 21] = adjust_using_previous_vector(right_ankle, previous_right_ankle, right_tip, previous_right_tip, threshold_tip)

        
        #smoothed_poses = kalman_filter_3d_with_joint_detection(pose_data_np)
        smoothed_poses = kalman_filter_3d_with_persistent_velocity(pose_data_np)


        for frame in range(1, smoothed_poses.shape[0]):
            left_ankle = smoothed_poses[frame, 6]  # Cheville gauche
            right_ankle = smoothed_poses[frame, 3]  # Cheville droite
            left_heel = smoothed_poses[frame, 31]  # Talon gauche
            right_heel = smoothed_poses[frame, 22]  # Talon droit
            left_tip = smoothed_poses[frame, 30]  # Pointe du pied gauche
            right_tip = smoothed_poses[frame, 21]  # Pointe du pied droite
            
            # Frame précédente
            previous_left_ankle = smoothed_poses[frame - 1, 6]
            previous_right_ankle = smoothed_poses[frame - 1, 3]
            previous_left_heel = smoothed_poses[frame - 1, 31]
            previous_right_heel = smoothed_poses[frame - 1, 22]
            previous_left_tip = smoothed_poses[frame - 1, 30]
            previous_right_tip = smoothed_poses[frame - 1, 21]

            # Ajuster talon gauche et droit
            smoothed_poses[frame, 31] = adjust_using_previous_vector(left_ankle, previous_left_ankle, left_heel, previous_left_heel, threshold_heel)
            smoothed_poses[frame, 22] = adjust_using_previous_vector(right_ankle, previous_right_ankle, right_heel, previous_right_heel, threshold_heel)
            
            # Ajuster pointe du pied gauche et droit
            smoothed_poses[frame, 30] = adjust_using_previous_vector(left_ankle, previous_left_ankle, left_tip, previous_left_tip, threshold_tip)
            smoothed_poses[frame, 21] = adjust_using_previous_vector(right_ankle, previous_right_ankle, right_tip, previous_right_tip, threshold_tip)

        before_kalman_metrics = evaluate_3d_metrics(before_kalman_pose, frame_rate, threshold_factor=5, jerk_threshold_factor=3)
        after_kalman_metric = evaluate_3d_metrics(smoothed_poses, frame_rate, threshold_factor=5, jerk_threshold_factor=3)
        percentage_metrics = calculate_percentage_difference_3d(before_kalman_metrics, after_kalman_metric)

        dict_metrics_3D['smoothness_mean']['before'].append(before_kalman_metrics['smoothness_mean'])
        dict_metrics_3D['smoothness_mean']['after'].append(after_kalman_metric['smoothness_mean'])
        dict_metrics_3D['smoothness_mean']['percentage_diff'].append(percentage_metrics['smoothness_mean'])

        dict_metrics_3D['smoothness_median']['before'].append(before_kalman_metrics['smoothness_median'])
        dict_metrics_3D['smoothness_median']['after'].append(after_kalman_metric['smoothness_median'])
        dict_metrics_3D['smoothness_median']['percentage_diff'].append(percentage_metrics['smoothness_median'])


        dict_metrics_3D['mean_velocity']['before'].append(before_kalman_metrics['mean_velocity'])
        dict_metrics_3D['mean_velocity']['after'].append(after_kalman_metric['mean_velocity'])
        dict_metrics_3D['mean_velocity']['percentage_diff'].append(percentage_metrics['mean_velocity'])

        dict_metrics_3D['std_velocity']['before'].append(before_kalman_metrics['std_velocity'])
        dict_metrics_3D['std_velocity']['after'].append(after_kalman_metric['std_velocity'])
        dict_metrics_3D['std_velocity']['percentage_diff'].append(percentage_metrics['std_velocity'])

        dict_metrics_3D['median_velocity']['before'].append(before_kalman_metrics['median_velocity'])
        dict_metrics_3D['median_velocity']['after'].append(after_kalman_metric['median_velocity'])
        dict_metrics_3D['median_velocity']['percentage_diff'].append(percentage_metrics['median_velocity'])

        dict_metrics_3D['velocity_flagged_frames']['before'].append(before_kalman_metrics['velocity_flagged_frames'])
        dict_metrics_3D['velocity_flagged_frames']['after'].append(after_kalman_metric['velocity_flagged_frames'])
        dict_metrics_3D['velocity_flagged_frames']['percentage_diff'].append(percentage_metrics['velocity_flagged_frames'])

        dict_metrics_3D['jerk_flagged_frames']['before'].append(before_kalman_metrics['jerk_flagged_frames'])
        dict_metrics_3D['jerk_flagged_frames']['after'].append(after_kalman_metric['jerk_flagged_frames'])
        dict_metrics_3D['jerk_flagged_frames']['percentage_diff'].append(percentage_metrics['jerk_flagged_frames'])

        dict_metrics_3D['median_jerk']['before'].append(before_kalman_metrics['median_jerk'])
        dict_metrics_3D['median_jerk']['after'].append(after_kalman_metric['median_jerk'])
        dict_metrics_3D['median_jerk']['percentage_diff'].append(percentage_metrics['median_jerk'])

        dict_metrics_3D['mean_jerk']['before'].append(before_kalman_metrics['mean_jerk'])
        dict_metrics_3D['mean_jerk']['after'].append(after_kalman_metric['mean_jerk'])
        dict_metrics_3D['mean_jerk']['percentage_diff'].append(percentage_metrics['mean_jerk'])



        # Update the original pose data with the smoothed results
        for i, frame in enumerate(segment):
            frames[incr]['j3d'] = np.round(smoothed_poses[i],3)
            incr += 1
    
    return frames, dict_metrics_3D




def kalman_filter_2d_pixel_with_persistent_velocity(joints_2D, frame_rate=60, process_noise=5e0, observation_noise=1e1, frame_step=5):
    """
    Apply a Kalman filter to smooth the 2D joint data in pixel space, including velocity estimation.
    The filter always uses the past velocity and retains it for future updates.
    
    Args:
    joints_2D: (N, 35, 2) array of joint positions over N frames.
    frame_rate: Frame rate of the video (frames per second), default is 60.
    process_noise: Process noise covariance scalar. Higher values mean the model allows more state changes.
    observation_noise: Observation noise covariance scalar. Higher values mean more trust in the Kalman correction.
    frame_step: Number of frames to compute velocity over, default is 5.
    
    Returns:
    smoothed_poses: Smoothed 2D joint data.
    """

    
    N, num_joints, _ = joints_2D.shape

    if N < frame_step:
        return joints_2D
    
    # Compute velocity over `frame_step` frames
    delta_t = frame_step / frame_rate  # time difference between frames considered for velocity
    velocity = (joints_2D[frame_step:] - joints_2D[:-frame_step]) / delta_t  # velocity estimation
    
    # Pad velocity to match the shape (assume no velocity for the first few frames)
    velocity = np.vstack([np.zeros((frame_step, num_joints, 2)), velocity])
    
    # Flatten position and velocity to (N, 35 * 2 * 2) -> first 35*2 for position, next 35*2 for velocity
    flat_poses = joints_2D.reshape(N, -1)
    flat_velocity = velocity.reshape(N, -1)
    
    # Stack position and velocity for each frame
    state_vector = np.hstack([flat_poses, flat_velocity])
    
    # Define transition matrix with velocity update
    dim = state_vector.shape[1] // 2  # Number of positions (e.g., 70 for 35 joints, 2D)
    transition_matrix = np.block([
        [np.eye(dim), delta_t * np.eye(dim)],  # Position update with velocity
        [np.zeros((dim, dim)), np.eye(dim)]   # Velocity remains constant
    ])
    
    # Define observation matrix (only observe positions, not velocity)
    observation_matrix = np.hstack([np.eye(dim), np.zeros((dim, dim))])
    
    # Process noise covariance matrix (for both position and velocity)
    process_covariance = process_noise * np.eye(state_vector.shape[1])
    
    # Observation noise covariance matrix (only for positions)
    observation_covariance = observation_noise * np.eye(dim)
    
    # Initialize the Kalman filter with the first frame as the initial mean
    kf = KalmanFilter(
        initial_state_mean=state_vector[0], 
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=process_covariance,
        observation_covariance=observation_covariance
    )

    # Initialize the covariance matrix for the first frame (using the process covariance)
    state_covariance = process_covariance

    smoothed_states = [state_vector[0]]

    # Apply Kalman filter with persistent velocity
    for i in range(1, N):
        # Use the previous frame's velocity without updating it
        state_vector[i, flat_poses.shape[1]:] = state_vector[i-1, flat_poses.shape[1]:]  # Retain previous velocity
        
        # Apply Kalman filter to smooth the data
        kf_state_mean, state_covariance = kf.filter_update(
            state_vector[i-1],  # previous state mean
            state_covariance,   # previous state covariance (required)
            observation=flat_poses[i]  # current observation
        )
        
        # Update the current state vector
        state_vector[i] = kf_state_mean
        smoothed_states.append(kf_state_mean)

    smoothed_states = np.array(smoothed_states)
    
    # Extract the smoothed positions (first 70 elements)
    smoothed_positions = smoothed_states[:, :flat_poses.shape[1]]
    
    # Reshape the smoothed positions back to original format (N, 35, 2)
    smoothed_poses = smoothed_positions.reshape(N, num_joints, 2)
    
    return smoothed_poses



def adjust_using_previous_vector_2d(current_ankle_pos, previous_ankle_pos, current_joint_pos, previous_joint_pos, threshold):
    """
    Adjust the current 2D joint position based on the previous frame's joint vector if the current vector deviates too much.
    
    Args:
    current_ankle_pos: The current position of the ankle (x, y).
    previous_ankle_pos: The previous position of the ankle (x, y).
    current_joint_pos: The current position of the joint to be adjusted (x, y).
    previous_joint_pos: The previous position of the joint (x, y).
    threshold: Distance threshold for deciding if adjustment is needed.
    
    Returns:
    adjusted_pos: The adjusted position of the joint (x, y) if needed, otherwise the current joint position.
    """
    # Compute the current 2D distance between the ankle and the joint
    distance = np.linalg.norm(current_joint_pos - current_ankle_pos)
    
    # If the distance is beyond the threshold, adjust using the previous vector
    if distance > threshold:
        # Calculate the previous frame's vector from ankle to joint
        previous_vector = previous_joint_pos - previous_ankle_pos
        
        # Apply the previous frame's vector to the current ankle position
        adjusted_pos = current_ankle_pos + previous_vector
        return adjusted_pos
    
    # Return the current joint position if no adjustment is needed
    return current_joint_pos


def apply_kalman_filter_to_2d_pose_data(frames, dict_metrics_2D, max_frame_gap=5):
    """
    Apply the Kalman filter to smooth 2D pose data for each player, ensuring that 
    Kalman filtering is applied only to consecutive frames. If the gap between frames
    exceeds `max_frame_gap`, we apply the filter on separate pose segments.
    
    Args:
    frames: The collected 2D pose data for each frame.
    max_frame_gap: Maximum allowable gap between frames before separating the data into segments.
    
    Returns:
    frames: Updated with smoothed 2D pose data.
    """
    
    # Initialize variables to store consecutive segments
    current_segment = []
    all_segments = []

    # Loop through the pose data and collect consecutive frame segments
    for i, frame in enumerate(frames):
        if i == 0:
            current_segment.append(frame)
        else:
            prev_frame_number = frames[i - 1]['frame_number']
            curr_frame_number = frame['frame_number']
            
            # If the gap between frames is too large, save the current segment and start a new one
            if curr_frame_number - prev_frame_number > max_frame_gap:
                all_segments.append(current_segment)
                current_segment = []
            
            current_segment.append(frame)
    
    # Append the last segment
    if current_segment:
        all_segments.append(current_segment)
    
    incr = 0
    threshold_heel = 15
    threshold_tip = 25
    frame_rate = 60
    # Apply the Kalman filter to each consecutive segment
    for segment in all_segments:
        if len(segment) == 1:
            print("Skipping Kalman filter for a single-frame segment")
            frames[incr]['j2d'] = segment[0]['j2d']
            incr += 1
            continue

        # Extract the 2D pose data for the current segment
        segment_poses = [frame['j2d'] for frame in segment]
        before_kalman_pose = np.array([frame['j2d'] for frame in segment])
        # Convert the pose data to a numpy array
        pose_data_np = np.array(segment_poses)

        for frame in range(1, pose_data_np.shape[0]):
            left_ankle = pose_data_np[frame, 6]  # Cheville gauche
            right_ankle = pose_data_np[frame, 3]  # Cheville droite
            left_heel = pose_data_np[frame, 31]  # Talon gauche
            right_heel = pose_data_np[frame, 22]  # Talon droit
            left_tip = pose_data_np[frame, 30]  # Pointe du pied gauche
            right_tip = pose_data_np[frame, 21]  # Pointe du pied droite

            # Frame précédente
            previous_left_ankle = pose_data_np[frame - 1, 6]
            previous_right_ankle = pose_data_np[frame - 1, 3]
            previous_left_heel = pose_data_np[frame - 1, 31]
            previous_right_heel = pose_data_np[frame - 1, 22]
            previous_left_tip = pose_data_np[frame - 1, 30]
            previous_right_tip = pose_data_np[frame - 1, 21]

            # Ajuster talon gauche et droit (heel adjustment)
            pose_data_np[frame, 31] = adjust_using_previous_vector_2d(left_ankle, previous_left_ankle, left_heel, previous_left_heel, threshold_heel)
            pose_data_np[frame, 22] = adjust_using_previous_vector_2d(right_ankle, previous_right_ankle, right_heel, previous_right_heel, threshold_heel)
            
            # Ajuster pointe du pied gauche et droit (tip of foot adjustment)
            pose_data_np[frame, 30] = adjust_using_previous_vector_2d(left_ankle, previous_left_ankle, left_tip, previous_left_tip, threshold_tip)
            pose_data_np[frame, 21] = adjust_using_previous_vector_2d(right_ankle, previous_right_ankle, right_tip, previous_right_tip, threshold_tip)
        #smoothed_poses = kalman_filter_2d_pixel(pose_data_np)
        smoothed_poses = kalman_filter_2d_pixel_with_persistent_velocity(pose_data_np)


        for frame in range(1, smoothed_poses.shape[0]):
            left_ankle = smoothed_poses[frame, 6]  # Cheville gauche
            right_ankle = smoothed_poses[frame, 3]  # Cheville droite
            left_heel = smoothed_poses[frame, 31]  # Talon gauche
            right_heel = smoothed_poses[frame, 22]  # Talon droit
            left_tip = smoothed_poses[frame, 30]  # Pointe du pied gauche
            right_tip = smoothed_poses[frame, 21]  # Pointe du pied droite

            # Frame précédente
            previous_left_ankle = smoothed_poses[frame - 1, 6]
            previous_right_ankle = smoothed_poses[frame - 1, 3]
            previous_left_heel = smoothed_poses[frame - 1, 31]
            previous_right_heel = smoothed_poses[frame - 1, 22]
            previous_left_tip = smoothed_poses[frame - 1, 30]
            previous_right_tip = smoothed_poses[frame - 1, 21]

            # Ajuster talon gauche et droit (heel adjustment)
            smoothed_poses[frame, 31] = adjust_using_previous_vector_2d(left_ankle, previous_left_ankle, left_heel, previous_left_heel, threshold_heel)
            smoothed_poses[frame, 22] = adjust_using_previous_vector_2d(right_ankle, previous_right_ankle, right_heel, previous_right_heel, threshold_heel)
            
            # Ajuster pointe du pied gauche et droit (tip of foot adjustment)
            smoothed_poses[frame, 30] = adjust_using_previous_vector_2d(left_ankle, previous_left_ankle, left_tip, previous_left_tip, threshold_tip)
            smoothed_poses[frame, 21] = adjust_using_previous_vector_2d(right_ankle, previous_right_ankle, right_tip, previous_right_tip, threshold_tip)

        before_kalman_metrics = evaluate_2d_metrics(before_kalman_pose, frame_rate, threshold_factor=5, jerk_threshold_factor=3)
        after_kalman_metric = evaluate_2d_metrics(smoothed_poses, frame_rate, threshold_factor=5, jerk_threshold_factor=3)

        percentage_metrics = calculate_percentage_difference_2d(before_kalman_metrics, after_kalman_metric)



        dict_metrics_2D['smoothness_mean']['before'].append(before_kalman_metrics['smoothness_mean'])
        dict_metrics_2D['smoothness_mean']['after'].append(after_kalman_metric['smoothness_mean'])
        dict_metrics_2D['smoothness_mean']['percentage_diff'].append(percentage_metrics['smoothness_mean'])

        dict_metrics_2D['smoothness_median']['before'].append(before_kalman_metrics['smoothness_median'])
        dict_metrics_2D['smoothness_median']['after'].append(after_kalman_metric['smoothness_median'])
        dict_metrics_2D['smoothness_median']['percentage_diff'].append(percentage_metrics['smoothness_median'])

        dict_metrics_2D['mean_velocity']['before'].append(before_kalman_metrics['mean_velocity'])
        dict_metrics_2D['mean_velocity']['after'].append(after_kalman_metric['mean_velocity'])
        dict_metrics_2D['mean_velocity']['percentage_diff'].append(percentage_metrics['mean_velocity'])

        dict_metrics_2D['std_velocity']['before'].append(before_kalman_metrics['std_velocity'])
        dict_metrics_2D['std_velocity']['after'].append(after_kalman_metric['std_velocity'])
        dict_metrics_2D['std_velocity']['percentage_diff'].append(percentage_metrics['std_velocity'])

        dict_metrics_2D['median_velocity']['before'].append(before_kalman_metrics['median_velocity'])
        dict_metrics_2D['median_velocity']['after'].append(after_kalman_metric['median_velocity'])
        dict_metrics_2D['median_velocity']['percentage_diff'].append(percentage_metrics['median_velocity'])

        dict_metrics_2D['velocity_flagged_frames']['before'].append(before_kalman_metrics['velocity_flagged_frames'])
        dict_metrics_2D['velocity_flagged_frames']['after'].append(after_kalman_metric['velocity_flagged_frames'])
        dict_metrics_2D['velocity_flagged_frames']['percentage_diff'].append(percentage_metrics['velocity_flagged_frames'])

        dict_metrics_2D['jerk_flagged_frames']['before'].append(before_kalman_metrics['jerk_flagged_frames'])
        dict_metrics_2D['jerk_flagged_frames']['after'].append(after_kalman_metric['jerk_flagged_frames'])
        dict_metrics_2D['jerk_flagged_frames']['percentage_diff'].append(percentage_metrics['jerk_flagged_frames'])

        dict_metrics_2D['median_jerk']['before'].append(before_kalman_metrics['median_jerk'])
        dict_metrics_2D['median_jerk']['after'].append(after_kalman_metric['median_jerk'])
        dict_metrics_2D['median_jerk']['percentage_diff'].append(percentage_metrics['median_jerk'])

        dict_metrics_2D['mean_jerk']['before'].append(before_kalman_metrics['mean_jerk'])
        dict_metrics_2D['mean_jerk']['after'].append(after_kalman_metric['mean_jerk'])
        dict_metrics_2D['mean_jerk']['percentage_diff'].append(percentage_metrics['mean_jerk'])

        # Update the original pose data with the smoothed results
        for i, frame in enumerate(segment):
            frames[incr]['j2d'] = np.round(smoothed_poses[i],3)
            incr += 1
    
    return frames, dict_metrics_2D


def update_poses_in_frames(data, clip_name):
    """
    Iterates through all players and their frames, loads the j3d and j2d data for each frame, 
    and updates the frame's pose based on the provided index_pose.
    
    Args:
    data: dict containing player data.
    clip_name: Name of the clip used to construct the file paths.
    
    Returns:
    None: Updates are made in-place to the data.
    """
    for player_id, player_info in data.items():
        print(f"Player {player_id} Pose Update ....")
        for frame in player_info['frames']:
            # Construct the path based on the clip name and frame number
            frame_number = str(frame['frame_number'])
            path_data = f'/n/home12/aandre/NBA-Players/results/{clip_name}/frame_{frame_number}/npy/'
            # Check if both j3d.npy and j2d.npy exist before loading
            if os.path.exists(path_data + 'j3d.npy') and os.path.exists(path_data + 'j2d.npy'):
                # Load the j3d and j2d numpy arrays
                j3d_frame = np.load(path_data + 'j3d.npy')  # [N, 35, 3]
                j2d_frame = np.load(path_data + 'j2d.npy')  # [N, 35, 3]

                # Make sure that the index_pose is within bounds of the loaded arrays
                index_pose = frame['index_pose']
                if index_pose < j3d_frame.shape[0] and index_pose < j2d_frame.shape[0]:
                    # Update the 'index_pose'-th pose in the frame with j2d and j3d
                    # frame["j2d"] = j2d_frame[index_pose]  # Update with the 2D pose
                    # frame["j3d"] = j3d_frame[index_pose]  # Update with the 3D pose
                    j2d_frame[index_pose]= frame["j2d"]  # Update with the 2D pose
                    j3d_frame[index_pose] = frame["j3d"]  # Update with the 3D pose
                    np.save(path_data + 'j2d.npy', j2d_frame)
                    np.save(path_data + 'j3d.npy', j3d_frame)
                else:
                    print(f"Index {index_pose} out of bounds for frame {frame_number}.")
            else:
                print(f"Files for frame {frame_number} not found in {path_data}")

def custom_serializer(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def main(clip_name, eval_bool, video_name):
    path_to_open = "players_data_test.json"

    # Load the player data from the JSON file
    with open(path_to_open) as f:
        data = json.load(f)

    dict_metrics_3D = {
        'smoothness_mean': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'smoothness_median': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'mean_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'std_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'median_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'median_jerk': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'mean_jerk': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'velocity_flagged_frames': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'jerk_flagged_frames': {
            'before': [],
            'after': [],
            'percentage_diff': []
        }
    }

    dict_metrics_2D = {
        'smoothness_mean': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'smoothness_median': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'mean_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'std_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'median_velocity': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'median_jerk': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'mean_jerk': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'velocity_flagged_frames': {
            'before': [],
            'after': [],
            'percentage_diff': []
        },
        'jerk_flagged_frames': {
            'before': [],
            'after': [],
            'percentage_diff': []
        }
    } 


    # Apply Kalman filter to 3D pose data
    for player_id, player_info in data.items():
        print("Processing 3D pose for player:", player_id)
        frames = player_info['frames']
        updated_frames, dict_metrics_3D = apply_kalman_filter_to_pose_data(frames, dict_metrics_3D)

        # Update the player's frames in the original data
        data[player_id]['frames'] = updated_frames

    # Apply Kalman filter to 2D pose data
    for player_id, player_info in data.items():
        print("Processing 2D pose for player:", player_id)
        frames = player_info['frames']
        updated_frames, dict_metrics_2D = apply_kalman_filter_to_2d_pose_data(frames, dict_metrics_2D)

        # Update the player's frames in the original data
        data[player_id]['frames'] = updated_frames


    print("\n" + "=" * 50)
    print("Average Metrics 3D Summary")
    print("=" * 50)

    for metric_name, values in dict_metrics_3D.items():
        # Filter out None values for before, after, and percentage_diff
        before_filtered = [v for v in values['before'] if v is not None]
        after_filtered = [v for v in values['after'] if v is not None]
        percentage_diff_filtered = [v for v in values['percentage_diff'] if v is not None]
        
        # Calculate averages or set to None if no valid values
        before_avg = sum(before_filtered) / len(before_filtered) if before_filtered else None
        after_avg = sum(after_filtered) / len(after_filtered) if after_filtered else None
        percentage_diff_avg = sum(percentage_diff_filtered) / len(percentage_diff_filtered) if percentage_diff_filtered else None
        
        # Calculate percentage difference between after_avg and before_avg
        if before_avg is not None and before_avg != 0:
            percentage_diff_avg_global = ((after_avg - before_avg) / abs(before_avg)) * 100 if after_avg is not None else None
        else:
            percentage_diff_avg_global = None  # Handle division by zero or if before_avg is None

        # Print in a formatted way
        print(f"\nMetric: {metric_name.capitalize()}")
        print(f"{'Before Average:':<20} {before_avg:.2f}" if before_avg is not None else f"{'Before Average:':<20} None")
        print(f"{'After Average:':<20} {after_avg:.2f}" if after_avg is not None else f"{'After Average:':<20} None")
        print(f"{'Percentage Global Diff:':<20} {percentage_diff_avg_global:.2f}%" if percentage_diff_avg_global is not None else f"{'Percentage Global Diff:':<20} None")
        print(f"{'Mean Percentage Diff Per Sample:':<20} {percentage_diff_avg:.2f}%" if percentage_diff_avg is not None else f"{'Mean Percentage Diff Per Sample:':<20} None")

    print("=" * 50 + "\n")


    print("\n" + "=" * 50)
    print("Average Metrics 2D Summary")
    print("=" * 50)

    for metric_name, values in dict_metrics_2D.items():
        # Filter out None values for before, after, and percentage_diff
        before_filtered = [v for v in values['before'] if v is not None]
        after_filtered = [v for v in values['after'] if v is not None]
        percentage_diff_filtered = [v for v in values['percentage_diff'] if v is not None]
        
        # Calculate averages or set to None if no valid values
        before_avg = sum(before_filtered) / len(before_filtered) if before_filtered else None
        after_avg = sum(after_filtered) / len(after_filtered) if after_filtered else None
        percentage_diff_avg = sum(percentage_diff_filtered) / len(percentage_diff_filtered) if percentage_diff_filtered else None
        
        # Calculate percentage difference between after_avg and before_avg
        if before_avg is not None and before_avg != 0:
            percentage_diff_avg_global = ((after_avg - before_avg) / abs(before_avg)) * 100 if after_avg is not None else None
        else:
            percentage_diff_avg_global = None  # Handle division by zero or if before_avg is None

        # Print in a formatted way
        print(f"\nMetric: {metric_name.capitalize()}")
        print(f"{'Before Average:':<20} {before_avg:.2f}" if before_avg is not None else f"{'Before Average:':<20} None")
        print(f"{'After Average:':<20} {after_avg:.2f}" if after_avg is not None else f"{'After Average:':<20} None")
        print(f"{'Percentage Global Diff:':<20} {percentage_diff_avg_global:.2f}%" if percentage_diff_avg_global is not None else f"{'Percentage Global Diff:':<20} None")
        print(f"{'Mean Percentage Diff Per Sample:':<20} {percentage_diff_avg:.2f}%" if percentage_diff_avg is not None else f"{'Mean Percentage Diff Per Sample:':<20} None")

    print("=" * 50 + "\n")




    # Update the poses in the frames based on the j3d and j2d data
    if eval_bool == True:
        # Save dict_metrics_3D to a JSON file
        output_file_3D = f'kalman_metrics/3D/{video_name}.json'
        with open(output_file_3D, 'w') as file_3D:
            json.dump(dict_metrics_3D, file_3D, indent=4)
        print(f"\n3D metrics summary saved to {output_file_3D}")

        # Save dict_metrics_2D to a JSON file
        output_file_2D = f'kalman_metrics/2D/{video_name}.json'
        with open(output_file_2D, 'w') as file_2D:
            json.dump(dict_metrics_2D, file_2D, indent=4)
        print(f"\n2D metrics summary saved to {output_file_2D}")




    else:
        update_poses_in_frames(data, clip_name)
        with open('data_test_after_kalman.json', 'w') as json_file:
            json.dump(data, json_file, indent=4, default=custom_serializer)
    print("Kalman filtering completed and saved to the original file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a clip file to extract player data.')
    parser.add_argument('--clip_name', type=str, help='Name of the clip.') 
    parser.add_argument('--eval', type=bool, default = False, help='Name of the clip.')
    parser.add_argument('--video_name', type=str, help='Name of the clip.')
    args = parser.parse_args()

    main(args.clip_name, args.eval, args.video_name)



