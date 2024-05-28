#!/bin/bash
### Unconditional ###

model_name_or_path="DaizeDong/GraphsGPT-1W"
generation_results_dir="./results/unconditional/generation/generated_results" # change this
save_dir="./results/unconditional/visualization"                              # change this
save_images="True"

# only visualize 2 files to save time & storage
file_begin_index=0
file_end_index=2

python entrypoints/generation/unconditional/visualize.py \
  --model_name_or_path ${model_name_or_path} \
  --generation_results_dir ${generation_results_dir} \
  --save_dir ${save_dir} \
  --save_images ${save_images} \
  --file_begin_index ${file_begin_index} \
  --file_end_index ${file_end_index}
