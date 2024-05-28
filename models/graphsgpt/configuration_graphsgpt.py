"""GraphsGPT Model Configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GraphsGPTConfig(PretrainedConfig):
    model_type = "graphs_gpt"

    def __init__(
            self,
            atom_vocab_size=118,  # number of atoms
            bond_vocab_size=92,  # number of bonds
            pad_token_id=0,
            share_embeddings=False,
            # --------------------- #
            node_loss_weight=1.0,
            connection_loss_weight=1.0,
            connection_loss_type="contrastive",  # classification contrastive
            adaptive_position_length=False,
            # --------------------- #
            num_fingerprints=8,
            position_feature_size=128,
            hidden_size=512,
            intermediate_size=2048,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_act="silu",
            # --------------------- #
            initializer_method="hidden",  # manual hidden hidden-layer
            initializer_range=0.02,  # useful only when "initializer_method" is "manual"
            rms_norm_eps=1e-6,
            gradient_checkpointing=False,
            **kwargs,
    ):
        self.atom_vocab_size = atom_vocab_size
        self.bond_vocab_size = bond_vocab_size
        self.share_embeddings = share_embeddings

        self.node_loss_weight = node_loss_weight
        self.connection_loss_weight = connection_loss_weight
        self.connection_loss_type = connection_loss_type
        self.adaptive_position_length = adaptive_position_length

        self.num_fingerprints = num_fingerprints
        self.position_feature_size = position_feature_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act

        self.initializer_method = initializer_method
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.gradient_checkpointing = gradient_checkpointing

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=False,
            # **kwargs,
        )
