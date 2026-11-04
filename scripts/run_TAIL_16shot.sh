#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=6

# Note: This order must be consistent with the order set in the 'dataset_sequence' parameter. 'dataset_sequence' is only used to construct the 'testloader'.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset aircraft num_epochs 40 continue_train_first True output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset caltech101 num_epochs 10 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset dtd num_epochs 30 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset eurosat num_epochs 100 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset flowers num_epochs 30 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset food101 num_epochs 5 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset mnist num_epochs 200 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset oxford_pets num_epochs 10 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset stanford_cars num_epochs 30 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m clip_vit_b16 num_shots 16 dataset sun397 num_epochs 10 output_dir TAIL_16shot

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python result_process.py -d TAIL --output_dir TAIL_16shot
