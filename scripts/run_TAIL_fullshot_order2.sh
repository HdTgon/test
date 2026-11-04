#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=6

# Note: This order must be consistent with the order set in the 'dataset_sequence' parameter. 'dataset_sequence' is only used to construct the 'testloader'.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset stanford_cars num_epochs 20 continue_train_first True output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset aircraft num_epochs 60 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset oxford_pets num_epochs 10 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset food101 num_epochs 10 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset sun397 num_epochs 2 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset mnist num_epochs 20 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset flowers num_epochs 40 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset dtd num_epochs 30 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset caltech101 num_epochs 20 output_dir TAIL_fullshot_order2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL_order2 -m clip_vit_b16 num_shots -1 dataset eurosat num_epochs 20 output_dir TAIL_fullshot_order2

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python result_process.py -d TAIL_order2 --output_dir TAIL_fullshot_order2
