import numpy as np
import pandas as pd
import json
from collections import defaultdict
import argparse



def calculate_features_kalman(player_id_clip, player_id_frame,frame, clip_name):
    """
    Placeholder function to calculate joint angles, angular velocity/acceleration, 
    and linear velocity/acceleration for the pelvis.
    
    You will need to implement your specific math for joint calculations here.
    """
    # Mocked data for joint angles and velocities.

    #path_data= '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/Input_Model/'+ match_data + action_name + '/frame_' + str(frame) +'/npy/'
    path_data = '/n/home12/aandre/NBA-Players/results/'+ clip_name + '/frame_' + str(frame) +'/npy/'
    j3d_frame = np.load(path_data + 'j3d.npy')
    j2d_frame = np.load(path_data + 'j2d.npy')
    
    img_path = path_data + 'img_paths.pkl'

    df_path = pd.read_pickle(img_path)

    # Extract frame numbers from the path strings
    frame_numbers = [int(path.split('_')[-1].split('.')[0]) for path in df_path]

    # Create a DataFrame with the index and corresponding frame number
    df_with_frames = pd.DataFrame({
        'index_frame': frame_numbers
    })
    index_pose = df_with_frames[df_with_frames['index_frame'] == player_id_frame].index[0]

    j3d = j3d_frame[index_pose]
    j2d = j2d_frame[index_pose]

    #trans_mat = np.array([[1.,0,0],[0,0,-1.],[0,1,0]])
    #j3d = np.dot(trans_mat,j3d.T).T

    return j2d, j3d, index_pose


def process_clip_file(clip_file_path, clip_name, first_frame, last_frame):
    players_data = defaultdict(lambda: {"player_id": None, "frames": []})
    count_players = {}
    
    with open(clip_file_path, 'r') as file:
        for line in file:

            data = line.strip().split(',')
            
            frame_number = int(data[0])
            if first_frame and last_frame:
                if (frame_number < int(first_frame)) or (frame_number > int(last_frame)):
                    continue

            player_id = int(data[1])

            if frame_number in count_players.keys():
                count_players[frame_number] += 1
            else:
                count_players[frame_number] = 1

            
            # Get number of players in the same frame (we will mock this).
            num_other_players = 9  
            # You can calculate this by counting distinct players in the same frame
            # Structure for one frame's data
            
            player_data = {
                "frame_number": frame_number,
                "num_other_players": num_other_players
            }
            
            # Calculate joint angles, velocities, accelerations
            j2d, j3d, index_pose = calculate_features_kalman(player_id, (count_players[frame_number] -1), frame_number, clip_name)
            
            # Add calculations to the player's frame data
            player_data["j2d"] = j2d.tolist()  
            player_data["j3d"] = j3d.tolist()  

            player_data["index_pose"] =  int(index_pose)

            
            # Store the data for the player
            players_data[player_id]["player_id"] = player_id
            players_data[player_id]["frames"].append(player_data)
    
    # Convert the defaultdict to a regular dict for JSON serialization
    players_data = {f"player_{player_id}": player_info for player_id, player_info in players_data.items()}
    
    return players_data



def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Process a clip file to extract player data.')
    parser.add_argument('--name_sequence', type=str, help='Name of the sequence to process.')
    parser.add_argument('--clip_name', type=str, help='Name of the clip.')
    parser.add_argument('--first_frame', type=str, help='first frame.')
    parser.add_argument('--last_frame', type=str, help='last frame.')
    args = parser.parse_args()


    clip_file_path = f'/n/home12/aandre/MixSortTracking/YOLOX_outputs/on_teste/track_results/{args.name_sequence}.txt'

    players_data = process_clip_file(clip_file_path, args.clip_name, args.first_frame, args.last_frame)

    # Save the data to a JSON file
    with open('/n/home12/aandre/NBA-Players/players_data_test.json', 'w') as file:
        json.dump(players_data, file, indent=4)
    
    print("Data saved to players_data.json")


if __name__ == "__main__":
    main()