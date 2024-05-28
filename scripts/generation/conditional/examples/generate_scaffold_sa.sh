#!/bin/bash

model_name_or_path="DaizeDong/GraphsGPT-1W-C"
save_dir="./results/conditional/generation_scaffold_sa" # change this
smiles_file="./data/examples/zinc_example.txt"
num_batches=10
batch_size=1024
seed=0

property_info_file="./configs/property_info.json"
value_sa=0.7               # change this
scaffold_smiles="c1ccccc1" # change this

strict_generation="False"
fix_aromatic_bond="True"
do_sample="False"
check_first_node="True"
check_atom_valence="True"

save_results="True"
save_failed="False"

python entrypoints/generation/conditional/generate.py \
  --model_name_or_path ${model_name_or_path} \
  --save_dir ${save_dir} \
  --smiles_file ${smiles_file} \
  --num_batches ${num_batches} \
  --batch_size ${batch_size} \
  --seed ${seed} \
  --property_info_file ${property_info_file} \
  --value_sa ${value_sa} \
  --scaffold_smiles ${scaffold_smiles} \
  --strict_generation ${strict_generation} \
  --do_sample ${do_sample} \
  --check_first_node ${check_first_node} \
  --check_atom_valence ${check_atom_valence} \
  --fix_aromatic_bond ${fix_aromatic_bond} \
  --save_results ${save_results} \
  --save_failed ${save_failed}
