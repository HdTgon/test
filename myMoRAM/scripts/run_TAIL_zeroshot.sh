#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=6

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python main.py -d TAIL -m zs_clip_vit_b16
