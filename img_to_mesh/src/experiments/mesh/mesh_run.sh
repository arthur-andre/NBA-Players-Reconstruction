#!/usr/bin/env sh
default_cfg_path=experiments/mesh/mesh_default_config.yaml
cfg_path=experiments/mesh/mesh_demo.yaml

python main.py \
    --default_cfg_path $default_cfg_path \
    --cfg_paths $cfg_path
