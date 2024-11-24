import numpy as np

def bring_coordinates_to_human_space_3d(poses_3D, person_height=1.90):
    """
    Scale the 3D body pose coordinates to human space, assuming a known height.
    
    Args:
    joints_3d: (N, 3) array of 3D joint coordinates.
    person_height: The real-world height of the person (in meters).
    
    Returns:
    scaled_joints_3d: (N, 3) array of 3D joint coordinates scaled to human space.
    """

    wrong_inference = False
    # Find the top-most and bottom-most y-coordinates (assuming y is up)
    top_z = poses_3D[34][2]
    bottom_z = np.min(poses_3D[:, 2])  # Bottom of the person (e.g., feet)

    
    # Calculate the current height of the person in 3D space
    current_height = np.abs(top_z - bottom_z)

    if current_height < 0.4:
        print("wrong inference with height = ",current_height)
        wrong_inference = True
        return None , wrong_inference
    
    # Calculate the scaling factor to bring the height to person_height meters
    scale_factor = person_height / current_height

    #print("Scale factor:", scale_factor)
    
    # Apply the scaling factor to all coordinates
    scaled_joints_3d = poses_3D * scale_factor
    
    return scaled_joints_3d, wrong_inference


def bring_coordinates_to_human_space_2d(joints_2d, person_height=1.90, pelvis_idx=0):
    """
    Scale the 2D body pose coordinates to human space and center around the pelvis.

    Args:
    joints_2d: (N, 2) array of 2D joint coordinates.
    person_height: The real-world height of the person (in meters).
    pelvis_idx: The index of the pelvis joint in the array.

    Returns:
    scaled_centered_joints_2d: (N, 2) array of 2D joint coordinates scaled and centered to human space.
    """

    # Center the joints around the pelvis
    pelvis_position = joints_2d[pelvis_idx]  # Get the pelvis coordinates
    scaled_centered_joints_2d = joints_2d - pelvis_position  # Subtract pelvis to center

    # Find the top-most and bottom-most y-coordinates (assuming y is the vertical axis)
    top_y = scaled_centered_joints_2d[34][1]  # Top of the person (e.g., head)
    bottom_y = np.max(scaled_centered_joints_2d[:, 1])  # Bottom of the person (e.g., feet)

    if bottom_y > top_y:
        current_height = np.abs(bottom_y - top_y)
    else:
        current_height = np.abs(top_y - bottom_y)
    

    if current_height == 0:
        raise ValueError("Current height is zero. Ensure valid joint coordinates.")
    
    # Calculate the scaling factor to bring the height to person_height meters
    scale_factor = person_height / current_height

    # Apply the scaling factor to all coordinates
    scaled_centered_joints_2d = scaled_centered_joints_2d * scale_factor

    

    return scaled_centered_joints_2d

def compute_smoothness_3d(poses):
    """
    Compute smoothness by calculating the mean and median absolute difference between consecutive frames for each joint in 3D.
    
    Args:
    poses: (N, M, 3) array of joint positions over N frames for M joints.
    
    Returns:
    tuple: Mean and median smoothness scores (absolute difference between consecutive frames).
    """
    poses_copy = np.copy(poses)  # Create a copy to avoid modifying the original
    N = poses_copy.shape[0]
    M = poses_copy.shape[1]
    trans_mat = np.array([[1., 0, 0],
                          [0, 0, -1.],
                          [0, 1, 0]])

    for i, poses_3D in enumerate(poses_copy):
        joints_3d_test = np.dot(trans_mat, poses_3D.T).T
        new_joints, wrong_inference = bring_coordinates_to_human_space_3d(joints_3d_test)
        if not wrong_inference:
            poses_copy[i, :, :] = new_joints
    
    all_diffs = []  # Collect all frame-by-frame differences
    
    for i in range(1, N):
        # Compute absolute difference between consecutive frames for each joint in 3D
        diff = np.linalg.norm(poses_copy[i] - poses_copy[i - 1], axis=1)
        all_diffs.extend(diff)
    
    # Compute mean of the absolute differences
    mean_smoothness = np.mean(all_diffs)
    
    # Compute median of the absolute differences
    median_smoothness = np.median(all_diffs)
    
    return mean_smoothness, median_smoothness

def compute_velocity_consistency_3d_one(poses, frame_rate):
    """
    Compute velocity consistency by calculating the velocity for each joint across frames in 3D.
    
    Args:
    poses: (N, M, 3) array of joint positions over N frames for M joints in 3D.
    frame_rate: Frame rate of the data (frames per second).
    
    Returns:
    tuple: Mean velocity and standard deviation of velocities for joints in 3D.
    """
    poses_copy = np.copy(poses)  # Create a copy to avoid modifying the original
    N = poses_copy.shape[0]
    M = poses_copy.shape[1]
    trans_mat = np.array([[1., 0, 0],
                          [0, 0, -1.],
                          [0, 1, 0]])

    for i, poses_3D in enumerate(poses_copy):
        joints_3d_test = np.dot(trans_mat, poses_3D.T).T
        new_joints, wrong_inference = bring_coordinates_to_human_space_3d(joints_3d_test)
        if not wrong_inference:
            poses_copy[i, :, :] = new_joints
    
    delta_t = 1 / frame_rate  # Time between frames
    velocities = np.zeros((N - 1, M, 3))
    
    for i in range(1, N):
        # Calculate velocity for each joint in 3D
        velocities[i - 1] = (poses_copy[i] - poses_copy[i - 1]) / delta_t
    
    # Calculate the velocity magnitude for each joint
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    
    # Compute mean and standard deviation of velocities
    mean_velocity = np.mean(velocity_magnitudes)
    median_velocity = np.median(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    
    return mean_velocity, std_velocity

def compute_velocity_consistency_3d(poses, frame_rate, min_time_frame=12):
    """
    Compute velocity consistency by calculating the velocity for each joint across frames in 3D.
    
    Args:
    poses: (N, M, 3) array of joint positions over N frames for M joints in 3D.
    frame_rate: Frame rate of the data (frames per second).
    min_time_frame: Minimum number of frames between positions to compute velocity (default is 12 frames apart).
    
    Returns:
    tuple: Mean velocity, median velocity, and standard deviation of velocities for joints in 3D.
    """
    poses_copy = np.copy(poses)  # Create a copy to avoid modifying the original
    N = poses_copy.shape[0]
    M = poses_copy.shape[1]
    trans_mat = np.array([[1., 0, 0],
                          [0, 0, -1.],
                          [0, 1, 0]])

    for i, poses_3D in enumerate(poses_copy):
        joints_3d_test = np.dot(trans_mat, poses_3D.T).T
        new_joints, wrong_inference = bring_coordinates_to_human_space_3d(joints_3d_test)
        if not wrong_inference:
            poses_copy[i, :, :] = new_joints
    
    if N <= min_time_frame:
        print("Not enough frames to compute velocities with the specified min_time_frame.")
        return None, None, None

    delta_t = min_time_frame / frame_rate  # Time interval between frames
    velocities = np.zeros((N - min_time_frame, M, 3))
    
    for i in range(min_time_frame, N):
        # Calculate velocity for each joint in 3D using 12 frames apart
        velocities[i - min_time_frame] = (poses_copy[i] - poses_copy[i - min_time_frame]) / delta_t
    
    # Calculate the velocity magnitude for each joint
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    
    # Compute mean, median, and standard deviation of velocities
    mean_velocity = np.mean(velocity_magnitudes)
    median_velocity = np.median(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    
    return mean_velocity, median_velocity, std_velocity

def compute_smoothness(smoothed_poses):
    """
    Compute smoothness by calculating the mean and median absolute difference between consecutive frames for each joint.
    
    Args:
    smoothed_poses: (N, M, 2) array of joint positions over N frames for M joints.
    
    Returns:
    tuple: Mean and median smoothness scores (absolute difference between consecutive frames).
    """
    smoothed_poses_copy = np.copy(smoothed_poses)  # Create a copy to avoid modifying the original
    N = smoothed_poses_copy.shape[0]
    M = smoothed_poses_copy.shape[1]

    for i, joints_2d in enumerate(smoothed_poses_copy):
        smoothed_poses_copy[i, :, :] = bring_coordinates_to_human_space_2d(joints_2d)
    
    all_diffs = []  # Collect all frame-by-frame differences
    
    for i in range(1, N):
        # Compute absolute difference between consecutive frames for each joint
        diff = np.linalg.norm(smoothed_poses_copy[i] - smoothed_poses_copy[i - 1], axis=1)
        all_diffs.extend(diff)
    
    # Compute mean of the absolute differences
    mean_smoothness = np.mean(all_diffs)
    
    # Compute median of the absolute differences
    median_smoothness = np.median(all_diffs)
    
    return mean_smoothness, median_smoothness

# Function to compute velocity consistency by calculating the difference between consecutive frames
def compute_velocity_consistency_one(smoothed_poses, frame_rate):
    """
    Compute velocity consistency by calculating the velocity for each joint across frames and comparing it with a 
    typical human movement range.
    
    Args:
    smoothed_poses: (N, M, 2) array of joint positions over N frames for M joints.
    frame_rate: Frame rate of the data (frames per second).
    
    Returns:
    tuple: Mean velocity and standard deviation of velocities for joints.
    """
    smoothed_poses_copy = np.copy(smoothed_poses)  # Create a copy to avoid modifying the original
    N = smoothed_poses_copy.shape[0]
    M = smoothed_poses_copy.shape[1]
    
    delta_t = 1 / frame_rate  # Time between frames
    velocities = np.zeros((N - 1, M, 2))
    
    for i in range(1, N):
        # Calculate velocity for each joint
        velocities[i - 1] = (smoothed_poses_copy[i] - smoothed_poses_copy[i - 1]) / delta_t
    
    # Calculate the velocity magnitude for each joint
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    
    # Compute mean and standard deviation of velocities
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    
    return mean_velocity, std_velocity

def compute_velocity_consistency_2d(smoothed_poses, frame_rate, min_time_frame=12):
    """
    Compute velocity consistency by calculating the velocity for each joint across frames in 2D.
    
    Args:
    smoothed_poses: (N, M, 2) array of joint positions over N frames for M joints.
    frame_rate: Frame rate of the data (frames per second).
    min_time_frame: Minimum number of frames between positions to compute velocity (default is 12 frames apart).
    
    Returns:
    tuple: Mean velocity, median velocity, and standard deviation of velocities for joints.
    """
    smoothed_poses_copy = np.copy(smoothed_poses)  # Create a copy to avoid modifying the original
    N = smoothed_poses_copy.shape[0]
    M = smoothed_poses_copy.shape[1]

    for i, joints_2d in enumerate(smoothed_poses_copy):
        smoothed_poses_copy[i, :, :] = bring_coordinates_to_human_space_2d(joints_2d)
    
    if N <= min_time_frame:
        print("Not enough frames to compute velocities with the specified min_time_frame.")
        return None, None, None

    delta_t = min_time_frame / frame_rate  # Time interval between frames
    velocities = np.zeros((N - min_time_frame, M, 2))
    
    for i in range(min_time_frame, N):
        # Calculate velocity for each joint using 12 frames apart
        velocities[i - min_time_frame] = (smoothed_poses_copy[i] - smoothed_poses_copy[i - min_time_frame]) / delta_t
    
    # Calculate the velocity magnitude for each joint
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    
    # Compute mean, median, and standard deviation of velocities
    mean_velocity = np.mean(velocity_magnitudes)
    median_velocity = np.median(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)
    
    return mean_velocity, median_velocity, std_velocity

# Function to calculate jerk for joint positions over time
def calculate_jerk_one(positions, frame_rate):
    """
    Calculate the jerk for joint positions over time.

    Args:
    positions: (N, num_joints, dim) array where N is the number of frames, 
               num_joints is the number of joints, and dim is 2 for 2D or 3 for 3D.
    frame_rate: Frame rate of the video (frames per second).

    Returns:
    jerk: (N-2, num_joints, dim) array representing the jerk for each joint.
    """
    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    delta_t = 1 / frame_rate
    velocity = np.diff(positions_copy, axis=0) / delta_t
    acceleration = np.diff(velocity, axis=0) / delta_t
    
    # Jerk is the rate of change of acceleration
    jerk = np.diff(acceleration, axis=0) / delta_t
    
    return jerk

def calculate_jerk(positions, frame_rate, min_time_frame=12):
    """
    Calculate the jerk for joint positions over time using a specified time frame difference.

    Args:
    positions: (N, num_joints, dim) array where N is the number of frames, 
               num_joints is the number of joints, and dim is 2 for 2D or 3 for 3D.
    frame_rate: Frame rate of the video (frames per second).
    min_time_frame: Minimum number of frames between positions to compute jerk (default is 12 frames apart).

    Returns:
    jerk: (N - 2 * min_time_frame, num_joints, dim) array representing the jerk for each joint.
    """
    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    delta_t = min_time_frame / frame_rate  # Time interval between frames

    # Compute velocity using min_time_frame difference
    velocity = (positions_copy[min_time_frame:] - positions_copy[:-min_time_frame]) / delta_t
    
    # Compute acceleration using min_time_frame difference
    acceleration = (velocity[min_time_frame:] - velocity[:-min_time_frame]) / delta_t
    
    # Compute jerk using min_time_frame difference
    jerk = (acceleration[min_time_frame:] - acceleration[:-min_time_frame]) / delta_t
    
    return jerk

# Function to apply jerk thresholding to identify abrupt changes in jerk values
def jerk_thresholding(positions, frame_rate, threshold_factor=3, min_time_frame=12):
    """
    Apply jerk thresholding to identify abrupt changes in jerk values.
    
    Args:
    positions: (N, num_joints, dim) array where N is the number of frames, 
               num_joints is the number of joints, and dim is 2 for 2D or 3 for 3D.
    frame_rate: Frame rate of the video (frames per second).
    threshold_factor: A multiplier to the standard deviation of jerk to set the threshold.
    
    Returns:
    tuple: A tuple containing:
        - flagged_frames: List of frames where the jerk change exceeds the threshold.
        - mean_jerk: Mean of the jerk magnitudes.
        - median_jerk: Median of the jerk magnitudes.
        - std_jerk: Standard deviation of the jerk magnitudes.
    """

    poses_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    N = poses_copy.shape[0]
    M = poses_copy.shape[1]
    dimension =poses_copy.shape[-1]

    if poses_copy.shape[0] <= 2 * min_time_frame:
        print("Not enough frames to compute velocities with the specified min_time_frame.")
        return None, None, None, None

    if dimension == 3:
        trans_mat = np.array([[1., 0, 0],
                            [0, 0, -1.],
                            [0, 1, 0]])

        for i, poses_3D in enumerate(poses_copy):
            joints_3d_test = np.dot(trans_mat, poses_3D.T).T
            new_joints, wrong_inference = bring_coordinates_to_human_space_3d(joints_3d_test)
            if not wrong_inference:
                poses_copy[i, :, :] = new_joints

    elif dimension == 2:
        for i, joints_2d in enumerate(poses_copy):
            poses_copy[i, :, :] = bring_coordinates_to_human_space_2d(joints_2d)

    else:
        print("wrong 3D or 2D input")

    jerk = calculate_jerk(poses_copy, frame_rate)

    # Calculate the jerk magnitude for each joint
    jerk_magnitudes = np.linalg.norm(jerk, axis=-1)

    # Calculate the mean, median, and standard deviation of jerk magnitudes
    mean_jerk = np.mean(jerk_magnitudes)
    median_jerk = np.median(jerk_magnitudes)
    std_jerk = np.std(jerk_magnitudes)

    # Define the jerk threshold
    jerk_threshold = mean_jerk + threshold_factor * std_jerk

    # Identify frames where the jerk change exceeds the threshold
    flagged_frames = []
    for i in range(1, len(jerk_magnitudes)):
        jerk_diff = np.abs(jerk_magnitudes[i] - jerk_magnitudes[i - 1])
        for j in range(len(jerk_diff)):
            if jerk_diff[j] > jerk_threshold:
                flagged_frames.append(i)
    
    return len(flagged_frames)*(100/N), mean_jerk, median_jerk, std_jerk

# Function to apply velocity thresholding for 2D data
def velocity_thresholding_2d_one(positions, frame_rate, threshold_factor=200):
    """
    Apply velocity thresholding for 2D data.

    Args:
    positions: (N, num_joints, 2) array of 2D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    threshold_factor: A multiplier to the standard deviation of velocity to set the threshold.

    Returns:
    flagged_frames: List of frames where the velocity exceeds the threshold.
    """
    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    delta_t = 1 / frame_rate
    # Compute velocity
    velocity = np.diff(positions_copy, axis=0) / delta_t

    # Calculate the mean and standard deviation of velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocity, axis=-1)
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)

    # Define the velocity threshold for 2D
    velocity_threshold = mean_velocity + threshold_factor * std_velocity

    # Flag frames where the velocity exceeds the threshold
    flagged_frames = np.where(velocity_magnitudes > velocity_threshold)[0]

    return flagged_frames


def velocity_thresholding_2d(positions, frame_rate, min_time_frame=12, threshold_factor=9):
    """
    Apply velocity thresholding for 2D data with a specified frame difference.

    Args:
    positions: (N, num_joints, 2) array of 2D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    min_time_frame: Minimum number of frames between positions to compute velocity (default is 12 frames apart).
    threshold_factor: A multiplier to the standard deviation of velocity to set the threshold.

    Returns:
    tuple: A tuple containing:
        - flagged_frames: List of frames where the velocity exceeds the threshold.
        - mean_velocity: Mean of the velocity magnitudes.
        - median_velocity: Median of the velocity magnitudes.
        - std_velocity: Standard deviation of the velocity magnitudes.
    """
    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    N = positions_copy.shape[0]

    for i, joints_2d in enumerate(positions_copy):
        positions_copy[i, :, :] = bring_coordinates_to_human_space_2d(joints_2d)

    if N <= min_time_frame:
        print("Not enough frames to compute velocities with the specified min_time_frame.")
        return None

    delta_t = min_time_frame / frame_rate  # Time interval between frames
    # Compute velocity with a frame difference of min_time_frame
    velocity = (positions_copy[min_time_frame:] - positions_copy[:-min_time_frame]) / delta_t

    # Calculate the mean, median, and standard deviation of velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocity, axis=-1)
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)

    # Define the velocity threshold for 2D
    velocity_threshold = mean_velocity + threshold_factor * std_velocity

    # Flag frames where the velocity exceeds the threshold
    flagged_frames = np.where(velocity_magnitudes > velocity_threshold)[0]

    return len(flagged_frames)*(100/N)

# Function to apply velocity thresholding for 3D data
def velocity_thresholding_3d_one(positions, frame_rate, threshold_factor=5):
    """
    Apply velocity thresholding for 3D data.

    Args:
    positions: (N, num_joints, 3) array of 3D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    threshold_factor: A multiplier to the standard deviation of velocity to set the threshold.

    Returns:
    flagged_frames: List of frames where the velocity exceeds the threshold.
    """

    N = positions.shape[0]

    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    delta_t = 1 / frame_rate
    # Compute velocity
    velocity = np.diff(positions_copy, axis=0) / delta_t

    # Calculate the mean and standard deviation of velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocity, axis=-1)
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)

    # Define the velocity threshold for 3D
    velocity_threshold = mean_velocity + threshold_factor * std_velocity

    # Flag frames where the velocity exceeds the threshold
    flagged_frames = np.where(velocity_magnitudes > velocity_threshold)[0]

    return len(flagged_frames)*(100/N)


def velocity_thresholding_3d(positions, frame_rate, min_time_frame=12, threshold_factor=9):
    """
    Apply velocity thresholding for 3D data with a specified frame difference.

    Args:
    positions: (N, num_joints, 3) array of 3D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    min_time_frame: Minimum number of frames between positions to compute velocity (default is 12 frames apart).
    threshold_factor: A multiplier to the standard deviation of velocity to set the threshold.

    Returns:
    tuple: A tuple containing:
        - flagged_frames: List of frames where the velocity exceeds the threshold.
        - mean_velocity: Mean of the velocity magnitudes.
        - median_velocity: Median of the velocity magnitudes.
        - std_velocity: Standard deviation of the velocity magnitudes.
    """
    positions_copy = np.copy(positions)  # Create a copy to avoid modifying the original
    N = positions_copy.shape[0]


    trans_mat = np.array([[1., 0, 0],
                        [0, 0, -1.],
                        [0, 1, 0]])

    for i, poses_3D in enumerate(positions_copy):
        joints_3d_test = np.dot(trans_mat, poses_3D.T).T
        new_joints, wrong_inference = bring_coordinates_to_human_space_3d(joints_3d_test)
        if not wrong_inference:
            positions_copy[i, :, :] = new_joints

    if N <= min_time_frame:
        print("Not enough frames to compute velocities with the specified min_time_frame.")
        return None

    delta_t = min_time_frame / frame_rate  # Time interval between frames
    # Compute velocity with a frame difference of min_time_frame
    velocity = (positions_copy[min_time_frame:] - positions_copy[:-min_time_frame]) / delta_t

    # Calculate the mean, median, and standard deviation of velocity magnitudes
    velocity_magnitudes = np.linalg.norm(velocity, axis=-1)
    mean_velocity = np.mean(velocity_magnitudes)
    std_velocity = np.std(velocity_magnitudes)

    # Define the velocity threshold for 3D
    velocity_threshold = mean_velocity + threshold_factor * std_velocity

    # Flag frames where the velocity exceeds the threshold
    flagged_frames = np.where(velocity_magnitudes > velocity_threshold)[0]

    return len(flagged_frames)*(100/N)


def evaluate_2d_metrics(joints_2d, frame_rate, threshold_factor=200, jerk_threshold_factor=3):
    """
    Evaluate 2D joint positions using various metrics including jerk thresholding.
    
    Args:
    joints_2d: (N, num_joints, 2) array of 2D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    threshold_factor: A multiplier to the standard deviation of velocity to set the velocity threshold.
    jerk_threshold_factor: A multiplier to the standard deviation of jerk to set the jerk threshold.
    
    Returns:
    dict: Dictionary containing smoothness, mean velocity, std velocity, jerk, and flagged frames.
    """
    smoothness_mean, smoothness_median = compute_smoothness(joints_2d)
    mean_velocity, median_velocity ,std_velocity = compute_velocity_consistency_2d(joints_2d, frame_rate)
    jerk = calculate_jerk(joints_2d, frame_rate)
    velocity_flagged_frames = velocity_thresholding_2d(joints_2d, frame_rate, threshold_factor)
    jerk_flagged_frames, mean_jerk, median_jerk, std_jerk = jerk_thresholding(joints_2d, frame_rate, jerk_threshold_factor)
    
    return {
        'smoothness_mean': smoothness_mean,
        'smoothness_median': smoothness_median,
        'mean_velocity': mean_velocity,
        'std_velocity': std_velocity,
        'median_velocity': median_velocity,
        'velocity_flagged_frames': velocity_flagged_frames,
        'jerk_flagged_frames': jerk_flagged_frames,
        'mean_jerk': mean_jerk,
        'median_jerk': median_jerk,
        'std_jerk': std_jerk
    }


def evaluate_3d_metrics(joints_3d, frame_rate, threshold_factor=5, jerk_threshold_factor=3):
    """
    Evaluate 3D joint positions using various metrics including jerk thresholding.
    
    Args:
    joints_3d: (N, num_joints, 3) array of 3D joint positions over time.
    frame_rate: Frame rate of the video (frames per second).
    threshold_factor: A multiplier to the standard deviation of velocity to set the velocity threshold.
    jerk_threshold_factor: A multiplier to the standard deviation of jerk to set the jerk threshold.
    
    Returns:
    dict: Dictionary containing smoothness, mean velocity, std velocity, jerk, and flagged frames.
    """
    smoothness_mean, smoothness_median = compute_smoothness_3d(joints_3d)
    mean_velocity, median_velocity, std_velocity = compute_velocity_consistency_3d(joints_3d, frame_rate)
    velocity_flagged_frames = velocity_thresholding_3d(joints_3d, frame_rate, threshold_factor)
    jerk_flagged_frames, mean_jerk, median_jerk, std_jerk = jerk_thresholding(joints_3d, frame_rate, jerk_threshold_factor)
    
    
    return {
        'smoothness_mean': smoothness_mean,
        'smoothness_median': smoothness_median,
        'mean_velocity': mean_velocity,
        'std_velocity': std_velocity,
        'median_velocity': median_velocity,
        'velocity_flagged_frames': velocity_flagged_frames,
        'jerk_flagged_frames': jerk_flagged_frames,
        'mean_jerk': mean_jerk,
        'median_jerk': median_jerk,
        'std_jerk': std_jerk
    }


def calculate_percentage_difference(before, after):
    """
    Calculate the percentage difference between two scalar values.
    
    Args:
    before (float): The metric value before applying changes.
    after (float): The metric value after applying changes.
    
    Returns:
    float: Percentage change.
    """
    if before == 0:
        return 0  # Return None or handle as appropriate if 'before' is zero

    if (before == None) or (after == None):
        return None
    
    percentage_change = ((after - before) / abs(before)) * 100
    return percentage_change


def calculate_percentage_difference_2d(before_metrics, after_metrics):
    """
    Calculate the percentage difference for each 2D metric between two sets of results.
    
    Args:
    before_metrics (dict): Dictionary of metrics before applying changes.
    after_metrics (dict): Dictionary of metrics after applying changes.
    
    Returns:
    list: List containing the percentage differences for smoothness, mean velocity, std velocity,
          velocity flagged frames, and jerk flagged frames.
    """
    # Calculate percentage differences for scalar metrics
    smoothness_mean_diff = calculate_percentage_difference(before_metrics['smoothness_mean'], after_metrics['smoothness_mean'])
    smoothness_median_diff = calculate_percentage_difference(before_metrics['smoothness_median'], after_metrics['smoothness_median'])
    mean_velocity_diff = calculate_percentage_difference(before_metrics['mean_velocity'], after_metrics['mean_velocity'])
    std_velocity_diff = calculate_percentage_difference(before_metrics['std_velocity'], after_metrics['std_velocity'])
    median_velocity_diff = calculate_percentage_difference(before_metrics['median_velocity'], after_metrics['median_velocity'])
    median_jerk_diff = calculate_percentage_difference(before_metrics['median_jerk'], after_metrics['median_jerk'])
    mean_jerk_diff = calculate_percentage_difference(before_metrics['mean_jerk'], after_metrics['mean_jerk'])
    # Calculate percentage differences for flagged frames based on counts
    velocity_flagged_diff = calculate_percentage_difference(
        before_metrics['velocity_flagged_frames'], after_metrics['velocity_flagged_frames']
    )
    jerk_flagged_diff = calculate_percentage_difference(
        before_metrics['jerk_flagged_frames'], after_metrics['jerk_flagged_frames']
    )

    # Return as a list
    return {
        'smoothness_mean': smoothness_mean_diff,
        'smoothness_median': smoothness_median_diff,
        'mean_velocity': mean_velocity_diff,
        'median_velocity': median_velocity_diff,
        'std_velocity': std_velocity_diff,
        'median_jerk': median_jerk_diff,
        'mean_jerk': mean_jerk_diff,
        'velocity_flagged_frames': velocity_flagged_diff,
        'jerk_flagged_frames': jerk_flagged_diff
    }


def calculate_percentage_difference_3d(before_metrics, after_metrics):
    """
    Calculate the percentage difference for each 3D metric between two sets of results.
    
    Args:
    before_metrics (dict): Dictionary of metrics before applying changes.
    after_metrics (dict): Dictionary of metrics after applying changes.
    
    Returns:
    list: List containing the percentage differences for smoothness, mean velocity, std velocity,
          velocity flagged frames, and jerk flagged frames.
    """
    # Calculate percentage differences for scalar metrics
    smoothness_mean_diff = calculate_percentage_difference(before_metrics['smoothness_mean'], after_metrics['smoothness_mean'])
    smoothness_median_diff = calculate_percentage_difference(before_metrics['smoothness_median'], after_metrics['smoothness_median'])
    mean_velocity_diff = calculate_percentage_difference(before_metrics['mean_velocity'], after_metrics['mean_velocity'])
    std_velocity_diff = calculate_percentage_difference(before_metrics['std_velocity'], after_metrics['std_velocity'])
    median_velocity_diff = calculate_percentage_difference(before_metrics['median_velocity'], after_metrics['median_velocity'])
    median_jerk_diff = calculate_percentage_difference(before_metrics['median_jerk'], after_metrics['median_jerk'])
    mean_jerk_diff = calculate_percentage_difference(before_metrics['mean_jerk'], after_metrics['mean_jerk'])

    # Calculate percentage differences for flagged frames based on counts
    velocity_flagged_diff = calculate_percentage_difference(
        before_metrics['velocity_flagged_frames'], after_metrics['velocity_flagged_frames']
    )
    jerk_flagged_diff = calculate_percentage_difference(
        before_metrics['jerk_flagged_frames'], after_metrics['jerk_flagged_frames']
    )

    # Return as a list
    return {
        'smoothness_mean': smoothness_mean_diff,
        'smoothness_median': smoothness_median_diff,
        'mean_velocity': mean_velocity_diff,
        'median_velocity': median_velocity_diff,
        'std_velocity': std_velocity_diff,
        'median_jerk': median_jerk_diff,
        'mean_jerk': mean_jerk_diff,
        'velocity_flagged_frames': velocity_flagged_diff,
        'jerk_flagged_frames': jerk_flagged_diff
    }