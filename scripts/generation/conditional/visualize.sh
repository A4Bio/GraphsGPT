#!/bin/bash
### Conditional ###

model_name_or_path="DaizeDong/GraphsGPT-1W-C"
generation_results_dir="./results/conditional/generation_scaffold_logp/generated_results" # change this
save_dir="./results/conditional/visualization_scaffold_logp"                              # change this
save_images="False"

# visualize all files to gather property info
file_begin_index=0
file_end_index=10

python entrypoints/generation/conditional/visualize.py \
  --model_name_or_path ${model_name_or_path} \
  --generation_results_dir ${generation_results_dir} \
  --save_dir ${save_dir} \
  --save_images ${save_images} \
  --file_begin_index ${file_begin_index} \
  --file_end_index ${file_end_index}
