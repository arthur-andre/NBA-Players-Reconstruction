#!/usr/bin/env sh

# Define the path to the YAML files
YAML_FILE_POSE="img_to_mesh/src/experiments/pose/pose_demo_model.yaml"

# Define new values
NEW_OUTPUT_NAME="28_bis"
NEW_SEQUENCE_NAME="clip"

# Use sed to replace the values in pose_demo.yaml
sed -i "s/^  output_name: .*/  output_name: $NEW_OUTPUT_NAME/" $YAML_FILE_POSE

echo "Options in $YAML_FILE_POSE have been updated."

# Change to the script directory
cd img_to_mesh/src

# Run the pose estimation
bash experiments/pose/pose_run_model.sh


#Run Kalman

cd ..
cd ..

python get_input_kalman.py --name_sequence $NEW_SEQUENCE_NAME --clip_name $NEW_OUTPUT_NAME

python update_poses_kalman.py --clip_name $NEW_OUTPUT_NAME
