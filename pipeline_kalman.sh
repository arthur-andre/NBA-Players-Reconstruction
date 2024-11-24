#!/usr/bin/env sh

# Define the path to the YAML files
YAML_FILE_POSE="img_to_mesh/src/experiments/pose/pose_demo.yaml"
YAML_FILE_MESH="img_to_mesh/src/experiments/mesh/mesh_demo.yaml"

# Define new values
NEW_STARTING_FRAME=1
NEW_ENDING_FRAME=120
NEW_OUTPUT_NAME="fultz"
NEW_SEQUENCE_NAME="fultz_injury"


# Use sed to replace the values in pose_demo.yaml
sed -i "s/^  starting_frame: .*/  starting_frame: $NEW_STARTING_FRAME/" $YAML_FILE_POSE
sed -i "s/^  ending_frame: .*/  ending_frame: $NEW_ENDING_FRAME/" $YAML_FILE_POSE
sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_POSE

echo "Options in $YAML_FILE_POSE have been updated."

# Use sed to replace the values in mesh_demo.yaml
sed -i "s/^  starting_frame: .*/  starting_frame: $NEW_STARTING_FRAME/" $YAML_FILE_MESH
sed -i "s/^  ending_frame: .*/  ending_frame: $NEW_ENDING_FRAME/" $YAML_FILE_MESH
sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_MESH

echo "Options in $YAML_FILE_MESH have been updated."


#python preprocess/MixSortPreprocess.py --out_fold_name fultz_2s --sequence_name injury_fultz


cd img_to_mesh/src

# Run the pose estimation
bash experiments/pose/pose_run.sh

#Run Kalman

cd ..
cd ..

python get_input_kalman.py --name_sequence $NEW_SEQUENCE_NAME --clip_name $NEW_OUTPUT_NAME --first_frame $NEW_STARTING_FRAME --last_frame $NEW_ENDING_FRAME

python update_poses_kalman.py --clip_name $NEW_OUTPUT_NAME

#Run the mesh model

cd img_to_mesh/src

bash experiments/mesh/mesh_run.sh