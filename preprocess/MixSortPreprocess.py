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

def center_crop(root_folder_img, next_input_folder, id_img, frame_path, seq_name):
    os.makedirs('{}/img_crop'.format(next_input_folder), exist_ok=True)
    os.makedirs('{}/proc_param'.format(next_input_folder), exist_ok=True)
    os.makedirs('{}/images'.format(next_input_folder), exist_ok=True)

    #NBA or on_teste = experience name, to be changed/adapted
    input_path_txt = '../MixSortTracking/YOLOX_outputs/on_teste/track_results/'+seq_name+'.txt' 
    players = get_lines_with_number(input_path_txt, id_img)
    frame_img = cv2.imread(frame_path)
    order_players = []
    for person_id , player in enumerate(players):
        parts = player.strip().split(',')
        order_players.append(f'player_{parts[1]}')
        x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        center = np.array([x1 + w/2, y1 + h/2])
        person_height = h
        scale = 150. / person_height 
        # crop image and keypoints
        img_crop, proc_param = scaleCrop_only(frame_img.copy(), scale, center, img_size=256)
        # save results
        cv2.imwrite('{}/img_crop/{}_{:04d}.png'.format(next_input_folder,'frame',person_id), img_crop)
        pickle.dump(proc_param, open('{}/proc_param/{}_{:04d}.pkl'.format(next_input_folder,'frame',person_id),'wb'))
        #raw image also saved

    order_players = np.array(order_players)

    np.save('{}/order_players_meshes.npy'.format(next_input_folder), order_players)
    cv2.imwrite('{}/images/frame.png'.format(next_input_folder), frame_img)



def main(ofn, seq_name):
    print("Start Preprocessing ....")
    #next_input_folder = '../NBA-Players/Input_NBA_model'
    next_input_folder_orig = 'Input_Model/' + ofn
    #root_folder_img = 'datasets/Test_Video/test/cavs_warriors/img1'
    root_folder_img = '../MixSortTracking/datasets/'+ofn+'/test/'+seq_name+'/img1'
    # Iterate over subfolders in Frames_NBA
    for index in range(len(os.listdir(root_folder_img))):
        frame_path = os.path.join(root_folder_img, f"{(index+1):06d}.jpg")
        #for each frame, create subfolder inside next input folder with the name of the frame 
        next_input_folder = f'{next_input_folder_orig}/frame_{index+1}'
        os.makedirs(next_input_folder, exist_ok=True)
        # Check if it's a directory
        if os.path.isfile(frame_path):
            if (index % 30) == 0:
                print(f"Processing frame : {index}")

            # Call center_crop function
            center_crop(root_folder_img, next_input_folder, index+1, frame_path, seq_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some frames.")
    parser.add_argument('--out_fold_name', type=str, required=True, help='The original output folder.')
    parser.add_argument('--sequence_name', type=str, required=True, help='The Sequence Name.')
    
    args = parser.parse_args()
    main(args.out_fold_name, args.sequence_name)
