#!/bin/bash

model_name_or_path="DaizeDong/GraphsGPT-1W"
save_path="./results/unconditional/moses" # change this

batch_size_valid=8196
sample_std=1.0
max_sample_times=10
num_shots=100000
num_samples_each_shot=1

num_processes=32

python entrypoints/generation/evaluation/evaluate_moses_few_shot_sampling.py \
  --model_name_or_path ${model_name_or_path} \
  --save_path ${save_path} \
  --batch_size_valid ${batch_size_valid} \
  --sample_std ${sample_std} \
  --max_sample_times ${max_sample_times} \
  --num_shots ${num_shots} \
  --num_samples_each_shot ${num_samples_each_shot} \
  --num_processes ${num_processes}
