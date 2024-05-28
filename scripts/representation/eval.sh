#!/bin/bash

python entrypoints/representation/finetune.py --task_name tox21
python entrypoints/representation/finetune.py --task_name toxcast
python entrypoints/representation/finetune.py --task_name bbbp
python entrypoints/representation/finetune.py --task_name sider
python entrypoints/representation/finetune.py --task_name hiv
python entrypoints/representation/finetune.py --task_name bace
