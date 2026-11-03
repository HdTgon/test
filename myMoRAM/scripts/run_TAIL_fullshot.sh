#!/bin/bash
set -e

# Note: This order must be consistent with the order set in the 'dataset_sequence' parameter. 'dataset_sequence' is only used to construct the 'testloader'.
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset aircraft continue_train_first True output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset caltech101 output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset dtd output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset eurosat output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset flowers output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset food101 output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset mnist output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset oxford_pets output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset stanford_cars output_dir TAIL_fullshot
python main.py -d TAIL -m moram_vit_b16 num_shots -1 dataset sun397 output_dir TAIL_fullshot

python result_process.py -d TAIL --output_dir TAIL_fullshot
