#!/bin/bash

model_name_or_path="DaizeDong/GraphsGPT-1W"
save_dir="./results/unconditional/generation" # change this
smiles_file="./data/examples/zinc_example.txt"
num_batches=10
batch_size=1024
seed=0

strict_generation="True"
fix_aromatic_bond="False"
do_sample="False"
check_first_node="True"
check_atom_valence="False"

save_results="True"
save_failed="False"

python entrypoints/generation/unconditional/generate.py \
  --model_name_or_path ${model_name_or_path} \
  --save_dir ${save_dir} \
  --smiles_file ${smiles_file} \
  --num_batches ${num_batches} \
  --batch_size ${batch_size} \
  --seed ${seed} \
  --strict_generation ${strict_generation} \
  --do_sample ${do_sample} \
  --check_first_node ${check_first_node} \
  --check_atom_valence ${check_atom_valence} \
  --fix_aromatic_bond ${fix_aromatic_bond} \
  --save_results ${save_results} \
  --save_failed ${save_failed}
