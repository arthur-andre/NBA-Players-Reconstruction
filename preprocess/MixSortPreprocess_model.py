import os
import os.path as osp
from glob import glob
import pickle
import json
import numpy as np
import cv2
import argparse
from utils import *
from PIL import Image
import shutil

def center_crop(root_folder_img, next_input_folder, id_img, frame_path, seq_name, match_name, action_name):
    os.makedirs('{}/img_crop'.format(next_input_folder), exist_ok=True)
    os.makedirs('{}/proc_param'.format(next_input_folder), exist_ok=True)
    #os.makedirs('{}/images'.format(next_input_folder), exist_ok=True)

    #NBA or on_teste = experience name, to be changed/adapted
    input_path_txt = '../MixSortTracking/YOLOX_outputs/'+match_name+ '/'+action_name+'/track_results/'+seq_name+'.txt' 
    players = get_lines_with_number(input_path_txt, id_img)
    frame_img = cv2.imread(frame_path)
    order_players = []
    for person_id , player in enumerate(players):
        parts = player.strip().split(',')
        order_players.append(f'player_{parts[1]}')
        x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        if (x1 < 0) or (y1 < 0):
            continue
        center = np.array([x1 + w/2, y1 + h/2])
        person_height = h
        scale = 150. / person_height 
        # crop image and keypoints
        img_crop, proc_param = scaleCrop_only(frame_img.copy(), scale, center, img_size=256)
        # save results
        cv2.imwrite('{}/img_crop/{}_{:04d}.png'.format(next_input_folder,'frame',person_id), img_crop)
        pickle.dump(proc_param, open('{}/proc_param/{}_{:04d}.pkl'.format(next_input_folder,'frame',person_id),'wb'))
        #raw image also saved
    np.save('{}/order_players_meshes.npy'.format(next_input_folder), order_players)
    #cv2.imwrite('{}/images/frame.png'.format(next_input_folder), frame_img)

def delete_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def main(match_name, seq_name, delete_folder_img = False):
    print("Start Preprocessing ....")
    next_input_folder_orig = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/Input_Model/' + match_name
    input_file = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/'+match_name+'/sorted_actions.json' 
    #input_file = '/n/home12/aandre/MixSortTracking/datasets/'+match_name+'/sorted_actions.json' 
    with open(input_file, 'r') as f:
        action_names = json.load(f)
    #directory_path = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/'+match_name
    print("Number of actions found : ", len(action_names))
    for action_name in action_names:
        action_name = str(action_name)
        print("We process the action : ", action_name)
        root_folder_img = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/'+match_name+'/' +action_name+'/test/'+seq_name+'/img1'
        #root_folder_img = '/n/home12/aandre/MixSortTracking/datasets/'+match_name+'/' +action_name+'/test/'+seq_name+'/img1'        
        print(len(os.listdir(root_folder_img)), ": number of frames inside action")
        # Iterate over subfolders in Frames_NBA
        for index in range(len(os.listdir(root_folder_img))):
            frame_path = os.path.join(root_folder_img, f"{(index+1):06d}.jpg")
            #for each frame, create subfolder inside next input folder with the name of the frame 
            next_input_folder = f'{next_input_folder_orig}/{action_name}/frame_{index+1}'
            os.makedirs(next_input_folder, exist_ok=True)
            # Check if it's a directory
            if os.path.isfile(frame_path):
                if (index % 30) == 0:
                    print(f"Processing frame : {index}")

                # Call center_crop function
                center_crop(root_folder_img, next_input_folder, index+1, frame_path, seq_name, match_name, action_name)
        if delete_folder_img:
            #folder_to_delete = '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/aandre/datasets/'+match_name+'/' +action_name
            folder_to_delete = '/n/home12/aandre/MixSortTracking/datasets/'+match_name+'/' +action_name
            delete_folder(folder_to_delete)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some frames.")
    parser.add_argument('--match_name', type=str, required=True, help='Match name.')
    parser.add_argument('--sequence_name', type=str, required=True, help='The Sequence Name.')
    parser.add_argument('--delete_folder', type=bool, default=False ,help='Boolean to force the folder removing to save storage.')
    
    args = parser.parse_args()
    if args.delete_folder: 
        main(args.match_name, args.sequence_name, args.delete_folder)
    else: 
        main(args.match_name, args.sequence_name)