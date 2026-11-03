#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=6

# Note: This order must be consistent with the order set in the 'dataset_sequence' parameter. 'dataset_sequence' is only used to construct the 'testloader'.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset aircraft continue_train_first True output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset caltech101 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset dtd output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset eurosat output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset flowers output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset food101 output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset mnist output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset oxford_pets output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset stanford_cars output_dir TAIL_16shot
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m moram_vit_b16 num_shots 16 dataset sun397 output_dir TAIL_16shot

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python result_process.py -d TAIL --output_dir TAIL_16shot
