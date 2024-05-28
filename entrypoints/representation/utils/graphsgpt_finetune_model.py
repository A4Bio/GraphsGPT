import torch
import torch.nn as nn

from data.tokenizer import GraphsGPTTokenizer
from models.graphsgpt.configuration_graphsgpt import GraphsGPTConfig
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM


class GraphsGPT(nn.Module):
    def __init__(self, task_name, mixup_strategy, pooler_dropout):
        super().__init__()
        if mixup_strategy == 'no_mix_vanilla':
            config = GraphsGPTConfig.from_pretrained("DaizeDong/GraphsGPT-1W")
            self.model = GraphsGPTForCausalLM(config)
        else:
            self.model = GraphsGPTForCausalLM.from_pretrained("DaizeDong/GraphsGPT-1W")
        self.tokenizer = GraphsGPTTokenizer.from_pretrained("DaizeDong/GraphsGPT-1W")

        if task_name in ['bace', 'bbbp', 'clintox', 'hiv']:
            num_classes = 2

        if task_name in ['esol', 'freesolv', 'lipo', 'qm7dft']:
            num_classes = 1

        if task_name in ['qm8dft']:
            num_classes = 12

        if task_name in ['qm9dft']:
            num_classes = 3

        if task_name in ['sider']:
            num_classes = 27

        if task_name in ['tox21']:
            num_classes = 12

        if task_name in ['toxcast']:
            num_classes = 617

        if task_name in ['muv']:
            num_classes = 17

        self.classification_heads = ClassificationHead(
            input_dim=512,
            inner_dim=512,
            num_classes=num_classes,
            pooler_dropout=pooler_dropout,
        )

    def forward(self, mols_list, mix_embeds=None, mixup_lam=None, mixup_index=None):
        batch = self.tokenizer.batch_encode(mols_list, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in batch.items()}
        fingerprint_tokens = self.model.encode_to_fingerprints(**inputs)
        logit_output = self.classification_heads(fingerprint_tokens)
        return {'logit_output': logit_output}


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = torch.tanh
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
