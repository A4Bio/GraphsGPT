# [GraphsGPT] A Graph is Worth $K$ Words:<br>Euclideanizing Graph using Pure Transformer (ICML2024)

**[Zhangyang Gao](https://scholar.google.com/citations?user=4SclT-QAAAAJ)\*, [Daize Dong](https://daizedong.github.io/)\*, [Cheng Tan](https://chengtan9907.github.io/), [Jun Xia](https://junxia97.github.io/), [Bozhen Hu](https://scholar.google.com/citations?user=6FZh9C8AAAAJ), [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ)**

Published on *The 41st International Conference on Machine Learning (ICML 2024)*.



![graphsgpt.svg](graphsgpt.svg)

This is the official code implementation of the ICML 2024 paper [A Graph is Worth $K$ Words: Euclideanizing Graph using Pure Transformer](https://arxiv.org/abs/2402.02464).

The model [checkpoints](https://huggingface.co/collections/DaizeDong/graphsgpt-65efe70c326a1a5bd35c2fcc) can be downloaded from 🤗 Transformers. We provide both the foundational pretrained models with different number of Graph Words $\mathcal{W}$ (GraphsGPT-$n$W), and the conditional version with one Graph Word (GraphsGPT-1W-C).

| Model Name     | 🤗 Checkpoint                                              |
| -------------- | --------------------------------------------------------- |
| GraphsGPT-1W   | https://huggingface.co/DaizeDong/GraphsGPT-1W/tree/main   |
| GraphsGPT-2W   | https://huggingface.co/DaizeDong/GraphsGPT-2W/tree/main   |
| GraphsGPT-4W   | https://huggingface.co/DaizeDong/GraphsGPT-4W/tree/main   |
| GraphsGPT-8W   | https://huggingface.co/DaizeDong/GraphsGPT-8W/tree/main   |
| GraphsGPT-1W-C | https://huggingface.co/DaizeDong/GraphsGPT-1W-C/tree/main |



## Installation

To get started with GraphsGPT, please run the following commands to install the environments.

```bash
git clone git@github.com:A4Bio/GraphsGPT.git
cd GraphsGPT
conda create --name graphsgpt python=3.12
conda activate graphsgpt
pip install -e .[dev]
pip install -r requirement.txt
```



## Quick Start

We provide some Jupyter Notebook examples in `./jupyter_notebooks`, you can run them for a quick start.

To use GraphsGPT as the pipeline, please refer to [example_pipeline.ipynb](jupyter_notebooks%2Fexample_pipeline.ipynb).



## Representation

You should first [download](https://github.com/A4Bio/GraphsGPT/releases/tag/data) the configurations and data for finetuning, and put them in `./data_finetune`. (We also include the finetuned checkpoints in the `model_zoom.zip` file for a quick test.)

To evaluate the representation performance of Graph2Seq Encoder, please run:

```bash
bash ./scripts/representation/finetune.sh
```

You can also toggle the `--mixup_strategy` for graph mixup using Graph2Seq.



## Generation

For unconditional generation with GraphGPT Decoder, please refer to [README-Generation-Uncond.md](scripts%2Fgeneration%2Funconditional%2FREADME-Generation-Uncond.md).

For conditional generation with GraphGPT-C Decoder, please refer to [README-Generation-Cond.md](scripts%2Fgeneration%2Fconditional%2FREADME-Generation-Cond.md).

To evaluate the few-shots generation performance of GraphGPT Decoder, please run:

```bash
bash ./scripts/generation/evaluation/moses.sh
bash ./scripts/generation/evaluation/zinc250k.sh
```



## Analysis

For further analysis on the Graph Words $\mathcal{W}$ of GraphsGPT (clustering, interpolation, and hybridization), please refer to the Jupyter Notebooks in `./jupyter_notebooks/analysis`.



## Citation

```latex
@article{gao2024graph,
  title={A Graph is Worth $ K $ Words: Euclideanizing Graph using Pure Transformer},
  author={Gao, Zhangyang and Dong, Daize and Tan, Cheng and Xia, Jun and Hu, Bozhen and Li, Stan Z},
  journal={arXiv preprint arXiv:2402.02464},
  year={2024}
}
```

## Contact Us
If you have any questions, please contact:

- Zhangyang Gao: gaozhangyang@westlake.edu.cn

- Daize Dong: dzdong2019@gmail.com
