""" PyTorch GraphsGPT model."""
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import warnings
from dataclasses import dataclass
from torch import nn, no_grad
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput
from typing import List, Optional, Tuple, Union

from data.collate_fn import tensor_stack_padding_collater
from models.graphsgpt.configuration_graphsgpt import GraphsGPTConfig
from models.graphsgpt.generation_utils import check_bond_connectivity_both_sides, check_bond_connectivity_begin, get_atom_ids_from_bond_id, get_another_atom_id_from_existing_bond, check_bond_in_graph, get_valence, VALENCE_LIMIT, fix_dissociative_aromatic_bond
from models.graphsgpt.orf import gaussian_orthogonal_random_matrix
from utils.accuracy import classification_accuracy
from utils.operations.operation_dict import reverse_dict
from utils.operations.operation_tensor import turn_last_true_mask_to_false, last_true_position

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GraphsGPTConfig"


@dataclass
class GraphPositionEmbeddingOutput(ModelOutput):
    graph_position_embeds: torch.FloatTensor = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None


@dataclass
class GraphsGPTEncoderOutput(ModelOutput):
    fingerprint_tokens: torch.FloatTensor = None

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None
    graph_position_embeds: Optional[torch.FloatTensor] = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None

    attention_mask: Optional[torch.Tensor] = None


@dataclass
class GraphsGPTDecoderOutputWithPast(ModelOutput):
    hidden_states: torch.FloatTensor = None

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None
    graph_position_embeds: Optional[torch.FloatTensor] = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None

    attention_mask: Optional[torch.Tensor] = None
    num_fingerprint_tokens: int = None

    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_query_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GraphsGPTModelOutputWithPast(ModelOutput):
    hidden_states: torch.FloatTensor = None

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None

    encoder_graph_position_embeds: Optional[torch.FloatTensor] = None
    encoder_graph_position_features: Optional[torch.FloatTensor] = None
    encoder_orthonormal_features: Optional[torch.FloatTensor] = None

    decoder_graph_position_embeds: Optional[torch.FloatTensor] = None
    decoder_graph_position_features: Optional[torch.FloatTensor] = None
    decoder_orthonormal_features: Optional[torch.FloatTensor] = None

    attention_mask: Optional[torch.Tensor] = None
    fingerprint_tokens: torch.FloatTensor = None
    num_fingerprint_tokens: int = None

    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_query_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GraphsGPTCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor = None
    loss_node: torch.FloatTensor = None
    loss_connection: torch.FloatTensor = None

    acc: Optional[torch.FloatTensor] = None
    acc_node: Optional[torch.FloatTensor] = None
    acc_connection_begin: Optional[torch.FloatTensor] = None
    acc_connection_end: Optional[torch.FloatTensor] = None

    non_padding_token_num: Optional[int] = None
    bond_num: Optional[int] = None
    fingerprint_tokens: torch.FloatTensor = None

    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_query_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
@no_grad()
def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str] = "cuda",
):
    """
    Make causal mask used for bidirectional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)  # shape(tgt_len,)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Modified from transformers.models.bart.modeling_bart._make_causal_mask
@no_grad()
def _make_graph_causal_mask(
        input_shape: torch.Size,
        identifier_ids: torch.BoolTensor,
        num_fingerprint_tokens: int,
        share_fingerprint_tokens: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Union[torch.device, str] = "cuda",
):
    """
    Make causal mask used for bidirectional self-attention with graph token.
    """
    # common lower triangular matrix mask
    bsz, tgt_len = input_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(tgt_len, device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(tgt_len, 1), 0)
    mask = mask.unsqueeze(0).repeat(bsz, 1, 1)

    # unmask before node tokens
    # A. the first node token should see the latter two tokens
    # B. the edge token before each node token should see the node token
    mole_seq_len = tgt_len - num_fingerprint_tokens - 1

    if mole_seq_len > 0:
        full_indices_mask = torch.arange(mole_seq_len, device=device).unsqueeze(0).expand(bsz, mole_seq_len)

        # (A) unmask for the first node token, it should see the latter two tokens
        # get unmask position indices for node tokens
        indices_batch = torch.arange(bsz, device=device).repeat_interleave(2, dim=0)
        indices_row = torch.full((2 * bsz,), num_fingerprint_tokens + 1, device=device)
        indices_column = torch.arange(num_fingerprint_tokens + 2, num_fingerprint_tokens + 4, step=1, device=device).unsqueeze(0).expand(bsz, 2).flatten()

        mask[indices_batch, indices_row, indices_column] = 0

        # (B) unmask before node tokens
        # make no change on the BOS token, i.e., the BOS token won't see the first node token
        identifier_ids = identifier_ids.clone()
        identifier_ids[:, 0] = False  # skip the first node token

        # get unmask position indices for node tokens
        node_num = identifier_ids.sum(dim=1)
        indices_batch = torch.arange(bsz, device=device).repeat_interleave(node_num, dim=0)
        indices_row = full_indices_mask[identifier_ids] + num_fingerprint_tokens
        indices_column = indices_row + 1

        mask[indices_batch, indices_row, indices_column] = 0

    # unmask within fingerprint tokens
    if share_fingerprint_tokens:
        mask[:, :, :num_fingerprint_tokens] = 0  # fingerprint tokens can see each other

    mask = mask.to(dtype)
    # print(mask[0, :, :].clone().cpu().numpy())

    return mask[:, None, :, :]  # (bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
@no_grad()
def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        tgt_len: int = None
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class StableEmbedding(nn.Embedding):
    """
    Stable embedding from https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/nn/modules.py#L21
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.norm = torch.nn.LayerNorm(embedding_dim)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None, per_sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        emb = emb.to(torch.get_default_dtype())  # always apply layer norm in full precision
        return self.norm(emb).to(self.weight.dtype)


class GraphPositionStableEmbedding(nn.Module):

    def __init__(self, feature_dim, embedding_dim) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        self.learnable_orthonormal_features = nn.Embedding(feature_dim, feature_dim)
        self.graph_position_proj = nn.Linear(2 * feature_dim, embedding_dim, bias=False)
        self.norm = torch.nn.LayerNorm(embedding_dim)

    def forward(
            self,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            dtype,
            device,
            use_random_id=False,
            adaptive_position_length=False,  # useful only when "use_random_id" is "True"
            orthonormal_features=None,
            graph_position_features=None,
            return_features=True,
    ) -> GraphPositionEmbeddingOutput:
        """
        Graph embedding modified from https://github.com/jw9730/tokengt
        Stable embedding from https://github.com/TimDettmers/bitsandbytes/blob/18e827d666fa2b70a12d539ccedc17aa51b2c97c/bitsandbytes/nn/modules.py#L21
        """
        if graph_position_features is None:
            batch_size, graph_seq_len = identifier_ids.shape  # also the shape of "graph_position_ids_1" and "graph_position_ids_2"
            max_node_cnt = int(torch.max(torch.sum(identifier_ids.clone(), dim=1)).item())

            # (batch_size, max_node_cnt, position_feature_size)
            if orthonormal_features is None:
                if use_random_id:  # randomly assign positional embeddings to atoms
                    if adaptive_position_length:  # indices range between (0, max_node_cnt)
                        _, embedding_ids = torch.rand((batch_size, max_node_cnt), device=device).sort(dim=1)  # random indices
                    else:  # indices range between (0, feature_dim)
                        _, embedding_ids = torch.rand((batch_size, self.feature_dim), device=device).sort(dim=1)  # random indices
                        embedding_ids = embedding_ids[:, :max_node_cnt]
                else:
                    embedding_ids = torch.arange(0, max_node_cnt, device=device).expand(batch_size, max_node_cnt)  # incremental indices
                orthonormal_features = self.learnable_orthonormal_features(embedding_ids)

            start_indices = torch.arange(0, batch_size * max_node_cnt, step=max_node_cnt, device=device).unsqueeze(1)  # (batch_size, 1)
            graph_position_features_1_indices = (graph_position_ids_1 + start_indices).reshape(-1)  # (batch_size * graph_seq_len)
            graph_position_features_2_indices = (graph_position_ids_2 + start_indices).reshape(-1)  # (batch_size * graph_seq_len)

            reshaped_orthonormal_features = orthonormal_features.reshape(-1, self.feature_dim)  # (batch_size * max_node_cnt, feature_dim)
            graph_position_features_1 = reshaped_orthonormal_features[graph_position_features_1_indices].reshape(batch_size, graph_seq_len, self.feature_dim)  # (batch_size, graph_seq_len, feature_dim)
            graph_position_features_2 = reshaped_orthonormal_features[graph_position_features_2_indices].reshape(batch_size, graph_seq_len, self.feature_dim)  # (batch_size, graph_seq_len, feature_dim)
            graph_position_features = torch.cat((graph_position_features_1, graph_position_features_2), dim=2)  # (batch_size, graph_seq_len, 2 * feature_dim)

        graph_position_embeds = self.graph_position_proj(graph_position_features)  # (batch_size, graph_seq_len, embedding_dim)
        graph_position_embeds = self.norm(graph_position_embeds).to(dtype)

        return GraphPositionEmbeddingOutput(
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features if return_features else None,
            orthonormal_features=orthonormal_features if return_features else None
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class GraphsGPTMLP(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
    ):
        super().__init__()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class GraphsGPTSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: GraphsGPTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_query_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()

        # TODO: the cache logic is bugged, need a fix
        if past_query_key_value is None:
            query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            past_seq_len = past_query_key_value[0].shape[-2]
            this_seq_len = seq_len - past_seq_len

            this_hidden_states = hidden_states[:, past_seq_len:, :]
            query_states = self.q_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = self.k_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value_states = self.v_proj(this_hidden_states).view(bsz, this_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # [bsz, nh, sl, hd]

            # reuse k, v, self_attention
            query_states = torch.cat([past_query_key_value[0], query_states], dim=2)
            key_states = torch.cat([past_query_key_value[1], key_states], dim=2)
            value_states = torch.cat([past_query_key_value[2], value_states], dim=2)

        past_query_key_value = (query_states, key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, seq_len, seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_query_key_value


class GraphsGPTEncoderLayer(nn.Module):
    def __init__(self, config: GraphsGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GraphsGPTSelfAttention(config=config)
        self.mlp = GraphsGPTMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)  # Pre-Normalization for Self Attention

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_query_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-Normalization for Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GraphsGPTDecoderLayer(nn.Module):
    def __init__(self, config: GraphsGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GraphsGPTSelfAttention(config=config)
        self.mlp = GraphsGPTMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_query_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_query_key_values` key value states are returned and can be used to speed up decoding
                (see `past_query_key_values`).
            past_query_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)  # Pre-Normalization for Self Attention

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_query_key_value=past_query_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)  # Pre-Normalization for Fully Connected
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GraphsGPTEncoder(nn.Module):
    def __init__(self, config: GraphsGPTConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.atom_vocab_size = config.atom_vocab_size
        self.bond_vocab_size = config.bond_vocab_size
        self.position_feature_size = config.position_feature_size

        # basic embeddings
        # 0 for padding token, 1 for B0S token
        # Although the encoder doesn't receive BOS token as the input, we still leave its position in case sharing embedding with the decoder.
        self.embed_tokens = StableEmbedding(2 + config.atom_vocab_size + config.bond_vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        # 0 for bonds (edges), 1 for atoms (nodes)
        self.embed_identifier = StableEmbedding(2, config.hidden_size)

        # fingerprint embeddings
        assert config.num_fingerprints > 0
        self.embed_fingerprint = StableEmbedding(1, config.hidden_size)
        self.embed_fingerprint_position = StableEmbedding(config.num_fingerprints, config.hidden_size)

        # graph position embeddings
        self.embed_graph_position = GraphPositionStableEmbedding(config.position_feature_size, config.hidden_size)

        # create layers
        self.encoder_layers = nn.ModuleList([GraphsGPTEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.encoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing

    def _prepare_encoder_attention_mask(
            self,
            attention_mask,
            input_shape,
            dtype=torch.float32,
            device="cuda",
    ):
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, dtype=dtype, tgt_len=input_shape[-1]).to(device)
        return expanded_attn_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            num_fingerprint_tokens: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> GraphsGPTEncoderOutput:

        batch_size, seq_length = identifier_ids.shape
        device = identifier_ids.device

        # input check
        if input_ids is not None and inputs_embeds is not None:
            assert input_ids.shape[:2] == inputs_embeds.shape[:2]  # ids and embeds must have the same length
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # input check
        if graph_position_ids_1 is not None and graph_position_ids_2 is not None:
            assert graph_position_ids_1.shape[:2] == graph_position_ids_2.shape[:2]  # both ids must have the same length
            if graph_position_embeds is not None:
                assert graph_position_ids_1.shape[:2] == graph_position_embeds.shape[:2]  # ids and embeds must have the same length
        elif graph_position_ids_1 is None and graph_position_ids_2 is None and graph_position_embeds is None:
            raise ValueError("You have to specify either graph_position_ids or graph_position_embeds")
        else:
            raise ValueError("graph_position_ids have to be either both specified or neither specified.")

        # set the number of encoded fingerprints
        if num_fingerprint_tokens is None:
            num_fingerprint_tokens = self.config.num_fingerprints

        # get encoder embeds
        fingerprint_embeds = self.embed_fingerprint(
            torch.zeros(batch_size, num_fingerprint_tokens, dtype=torch.int64, device=device)
        )  # (batch_size, num_fingerprint_tokens, embed_dim)
        fingerprint_position_embeds = self.embed_fingerprint_position(
            torch.arange(0, num_fingerprint_tokens, device=device).expand(batch_size, num_fingerprint_tokens)
        )  # (batch_size, num_fingerprint_tokens, embed_dim)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch_size, seq_len, embed_dim)
        dtype = inputs_embeds.dtype

        if graph_position_embeds is None:
            graph_embedding_outputs: GraphPositionEmbeddingOutput = self.embed_graph_position(
                graph_position_ids_1,
                graph_position_ids_2,
                identifier_ids,
                dtype=dtype,
                device=device,
                use_random_id=True,
                adaptive_position_length=self.config.adaptive_position_length,
                orthonormal_features=orthonormal_features,
                graph_position_features=graph_position_features,
                return_features=False,
            )
            graph_position_embeds = graph_embedding_outputs.graph_position_embeds  # (batch_size, seq_len, embed_dim)
            graph_position_features = graph_embedding_outputs.graph_position_features  # None
            orthonormal_features = graph_embedding_outputs.orthonormal_features  # None

        if identifier_embeds is None:
            identifier_embeds = self.embed_identifier(identifier_ids.clone().int())  # (batch_size, seq_len, embed_dim)

        # add embeds together and get hidden_states
        fingerprint_tokens = fingerprint_embeds + fingerprint_position_embeds  # (batch_size, num_fingerprint_tokens, embed_dim)
        molecule_tokens = inputs_embeds + graph_position_embeds + identifier_embeds  # (batch_size, seq_len, embed_dim)

        hidden_states = torch.cat((fingerprint_tokens, molecule_tokens), dim=1)  # (batch_size, num_fingerprint_tokens + seq_len, embed_dim)

        # get attention masks
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, num_fingerprint_tokens + seq_length), dtype=torch.bool, device=device)
        else:  # the attention mask is shaped like (batch_size, mole_seq_len) so we need to extend its dimension
            extra_dim = num_fingerprint_tokens + seq_length - attention_mask.shape[1]
            if extra_dim > 0:  # adding extra dimensions to the attention mask
                extra_attention_mask = torch.ones((batch_size, extra_dim), dtype=torch.bool, device=device)
                attention_mask = torch.cat((extra_attention_mask, attention_mask), dim=1)
            else:
                attention_mask = attention_mask

        attention_mask = self._prepare_encoder_attention_mask(
            attention_mask,
            (batch_size, num_fingerprint_tokens + seq_length),
            dtype=dtype,
            device=device,
        )

        # forward encoder
        for idx, encoder_layer in enumerate(self.encoder_layers):
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                )

        hidden_states = self.encoder_norm(hidden_states)  # (batch_size, num_fingerprint_tokens + seq_len, embed_dim)

        # get encoded fingerprint tokens
        fingerprint_tokens = hidden_states[:, :num_fingerprint_tokens, :]  # (batch_size, num_fingerprint_tokens, embed_dim)

        return GraphsGPTEncoderOutput(
            fingerprint_tokens=fingerprint_tokens,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            attention_mask=attention_mask,
        )


class GraphsGPTDecoder(nn.Module):
    def __init__(
            self,
            config: GraphsGPTConfig,
            embed_tokens: Optional[StableEmbedding] = None,
            embed_identifier: Optional[StableEmbedding] = None,
            embed_fingerprint_position: Optional[StableEmbedding] = None,
            embed_graph_position: Optional[GraphPositionStableEmbedding] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.atom_vocab_size = config.atom_vocab_size
        self.bond_vocab_size = config.bond_vocab_size
        self.position_feature_size = config.position_feature_size
        self.causal_mask_type = "graph"  # standard graph

        # basic embeddings
        # 0 for padding token, 1 for B0S token, 2-119 for atoms, 120-? for bonds
        self.embed_tokens = StableEmbedding(2 + config.atom_vocab_size + config.bond_vocab_size, config.hidden_size, padding_idx=self.padding_idx) if embed_tokens is None else embed_tokens
        # 0 for bonds (edges), 1 for atoms (nodes)
        self.embed_identifier = StableEmbedding(2, config.hidden_size) if embed_identifier is None else embed_identifier

        # fingerprint embeddings
        if config.num_fingerprints > 0:
            self.embed_fingerprint_position = StableEmbedding(config.num_fingerprints, config.hidden_size) if embed_fingerprint_position is None else embed_fingerprint_position

        # graph position embeddings
        self.embed_graph_position = GraphPositionStableEmbedding(config.position_feature_size, config.hidden_size) if embed_graph_position is None else embed_graph_position

        # create layers
        self.decoder_layers = nn.ModuleList([GraphsGPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.decoder_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = config.gradient_checkpointing

    # Modified from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(
            self,
            attention_mask,
            input_shape,
            identifier_ids,
            num_fingerprint_tokens,
            share_fingerprint_tokens=True,
            dtype=torch.float32,
            device="cuda",
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:  # seq_len > 1
            if self.causal_mask_type == "standard":
                combined_attention_mask = _make_causal_mask(
                    input_shape,
                    dtype=dtype,
                    device=device,
                )
            elif self.causal_mask_type == "graph":
                combined_attention_mask = _make_graph_causal_mask(
                    input_shape,
                    identifier_ids,
                    num_fingerprint_tokens,
                    share_fingerprint_tokens=share_fingerprint_tokens,
                    dtype=dtype,
                    device=device,
                )
            else:
                raise NotImplementedError

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
            combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask)

        return combined_attention_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            fingerprint_tokens: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,

            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            past_query_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> GraphsGPTDecoderOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        batch_size, mole_seq_len = identifier_ids.shape  # mole_seq_len can be 0, which means the input contains only BOS token
        device = identifier_ids.device

        # input check
        if input_ids is not None and inputs_embeds is not None:
            assert input_ids.shape[:2] == inputs_embeds.shape[:2]  # ids and embeds must have the same length
            input_seq_len = input_ids.shape[1]
        elif input_ids is not None:
            input_seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            input_seq_len = inputs_embeds.shape[1]
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        assert input_seq_len == mole_seq_len + 1  # BOS token in input_ids/inputs_embeds

        # input check
        if graph_position_ids_1 is not None and graph_position_ids_2 is not None:
            assert graph_position_ids_1.shape[:2] == graph_position_ids_2.shape[:2]  # both ids must have the same length
            if graph_position_embeds is not None:
                assert graph_position_ids_1.shape[:2] == graph_position_embeds.shape[:2]  # ids and embeds must have the same length
        elif graph_position_ids_1 is None and graph_position_ids_2 is None and graph_position_embeds is None:
            raise ValueError("You have to specify either graph_position_ids or graph_position_embeds")
        else:
            raise ValueError("graph_position_ids have to be either both specified or neither specified.")

        # get decoder embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # (batch_size, 1 + mole_seq_len, embed_dim)
        dtype = inputs_embeds.dtype

        if mole_seq_len > 0:
            if graph_position_embeds is None or graph_position_features is None:
                graph_embedding_outputs: GraphPositionEmbeddingOutput = self.embed_graph_position(
                    graph_position_ids_1,
                    graph_position_ids_2,
                    identifier_ids,
                    dtype=dtype,
                    device=device,
                    use_random_id=False,
                    adaptive_position_length=False,
                    graph_position_features=graph_position_features,
                    orthonormal_features=orthonormal_features,
                    return_features=True,
                )
                graph_position_embeds = graph_embedding_outputs.graph_position_embeds  # (batch_size, mole_seq_len, embed_dim)
                graph_position_features = graph_embedding_outputs.graph_position_features  # (batch_size, mole_seq_len, 2 * position_feature_size)
                orthonormal_features = graph_embedding_outputs.orthonormal_features  # (batch_size, max_atom_num, position_feature_size)

            if identifier_embeds is None:
                identifier_embeds = self.embed_identifier(identifier_ids.clone().int())  # (batch_size, mole_seq_len, embed_dim)
        else:
            graph_position_embeds = torch.empty((batch_size, 0, self.config.hidden_size), dtype=dtype, device=device)
            graph_position_features = torch.empty((batch_size, 0, 2 * self.config.position_feature_size), dtype=dtype, device=device)
            orthonormal_features = torch.empty((batch_size, 0, self.config.position_feature_size), dtype=dtype, device=device)
            identifier_embeds = torch.empty((batch_size, 0, self.config.hidden_size), dtype=dtype, device=device)

        # add embeds together and get hidden_states
        molecule_tokens = inputs_embeds  # (batch_size, 1 + mole_seq_len, embed_dim)
        if mole_seq_len > 0:  # add positions & identifiers
            molecule_tokens[:, -mole_seq_len:, :] += graph_position_embeds + identifier_embeds  # ignore the BOS token
        hidden_states = molecule_tokens  # (batch_size, 1 + mole_seq_len, embed_dim)

        if fingerprint_tokens is None:
            num_fingerprint_tokens = 0
        else:
            num_fingerprint_tokens = fingerprint_tokens.shape[1]
            fingerprint_position_embeds = self.embed_fingerprint_position(
                torch.arange(0, num_fingerprint_tokens, device=device).expand(batch_size, num_fingerprint_tokens)
            )  # (batch_size, num_fingerprint_tokens, embed_dim)
            hidden_states = torch.cat((fingerprint_tokens + fingerprint_position_embeds, hidden_states), dim=1)  # (batch_size, num_fingerprint_tokens + 1 + mole_seq_len, embed_dim)

        # past values
        seq_len = num_fingerprint_tokens + 1 + mole_seq_len

        # get attention masks
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        else:  # the attention mask is shaped like (batch_size, mole_seq_len) so we need to extend its dimension
            extra_dim = seq_len - attention_mask.shape[1]
            if extra_dim > 0:  # adding extra dimensions to the attention mask
                extra_attention_mask = torch.ones((batch_size, extra_dim), dtype=torch.bool, device=device)
                attention_mask = torch.cat((extra_attention_mask, attention_mask), dim=1)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_len),
            identifier_ids,
            num_fingerprint_tokens,
            share_fingerprint_tokens=True,
            dtype=dtype,
            device=device,
        )

        # cache check
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # forward decoder
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_query_key_value = past_query_key_values[idx] if past_query_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_query_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_query_key_value=past_query_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if use_cache:
                next_cache += (layer_outputs[2 if output_attentions else 1],)

        hidden_states = self.decoder_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_cache if use_cache else None

        return GraphsGPTDecoderOutputWithPast(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            attention_mask=attention_mask,
            num_fingerprint_tokens=num_fingerprint_tokens,
            all_hidden_states=all_hidden_states,
            all_attentions=all_self_attns,
            past_query_key_values=next_cache,
        )


class GraphsGPTPreTrainedModel(PreTrainedModel):
    config_class = GraphsGPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraphsGPTEncoderLayer", "GraphsGPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        # Init std from https://github.com/bigscience-workshop/bigscience/blob/master/train/lessons-learned.md
        if self.config.initializer_method == "manual":
            std = self.config.initializer_range
        elif self.config.initializer_method == "hidden":
            std = math.sqrt(2 / (self.config.hidden_size + self.config.intermediate_size))
        elif self.config.initializer_method == "hidden-layer":
            std = math.sqrt(2 / (self.config.hidden_size + self.config.intermediate_size)) / math.sqrt(2.0 * self.config.num_hidden_layers)
        else:
            raise NotImplementedError

        # print(f"Initializer range is {std}.")

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, GraphPositionStableEmbedding):
            # orthonormal initialization for graph position embeddings
            module.learnable_orthonormal_features.weight.data = gaussian_orthogonal_random_matrix(
                self.config.position_feature_size,
                self.config.position_feature_size,
                random_shuffle=False,
                device=module.learnable_orthonormal_features.weight.device,
                dtype=module.learnable_orthonormal_features.weight.dtype,
            )
            module.learnable_orthonormal_features._is_hf_initialized = True  # set to True to prevent override

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GraphsGPTModel):
            module.gradient_checkpointing = value


class GraphsGPTModel(GraphsGPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GraphsGPTDecoderLayer`]

    Args:
        config: GraphsGPTConfig
    """

    def __init__(self, config: GraphsGPTConfig):
        super().__init__(config)
        if config.num_fingerprints > 0:  # encoder-decoder
            self.encoder = GraphsGPTEncoder(config)
            self.decoder = GraphsGPTDecoder(
                config,
                embed_tokens=None if not config.share_embeddings else self.encoder.embed_tokens,
                embed_identifier=None if not config.share_embeddings else self.encoder.embed_identifier,
                embed_fingerprint_position=None if not config.share_embeddings else self.encoder.embed_fingerprint_position,
                embed_graph_position=None if not config.share_embeddings else self.encoder.embed_graph_position,
            )
        else:  # decoder only
            self.decoder = GraphsGPTDecoder(config)

        self.post_init()  # Initialize weights and apply final processing

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            # ↓↓↓↓ for encoder layers ↓↓↓↓
            encoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_identifier_embeds: Optional[torch.FloatTensor] = None,
            encoder_graph_position_embeds: Optional[torch.FloatTensor] = None,
            encoder_graph_position_features: Optional[torch.FloatTensor] = None,
            encoder_orthonormal_features: Optional[torch.FloatTensor] = None,

            # ↓↓↓↓ for decoder layers ↓↓↓↓
            fingerprint_tokens: Optional[torch.Tensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_identifier_embeds: Optional[torch.FloatTensor] = None,
            decoder_graph_position_embeds: Optional[torch.FloatTensor] = None,
            decoder_graph_position_features: Optional[torch.FloatTensor] = None,
            decoder_orthonormal_features: Optional[torch.FloatTensor] = None,

            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            past_query_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> GraphsGPTModelOutputWithPast:
        """forward function for end-to-end model training"""
        encoder_input_ids = input_ids
        decoder_input_ids = torch.cat((
            torch.ones((input_ids.shape[0], 1), dtype=torch.int64, device=input_ids.device),
            input_ids
        ), dim=1)  # add BOS token on the left side

        if fingerprint_tokens is None:
            num_fingerprint_tokens = self.config.num_fingerprints
        else:
            num_fingerprint_tokens = fingerprint_tokens.shape[1]

        # encoder
        if num_fingerprint_tokens <= 0:
            fingerprint_tokens = None
        else:
            encoder_outputs: GraphsGPTEncoderOutput = self.encoder(
                encoder_input_ids,
                graph_position_ids_1,
                graph_position_ids_2,
                identifier_ids,
                num_fingerprint_tokens=num_fingerprint_tokens,
                attention_mask=attention_mask,
                inputs_embeds=encoder_inputs_embeds,
                identifier_embeds=encoder_identifier_embeds,
                graph_position_embeds=encoder_graph_position_embeds,
                graph_position_features=encoder_graph_position_features,
                orthonormal_features=encoder_orthonormal_features,
                **kwargs,
            )
            fingerprint_tokens = encoder_outputs.fingerprint_tokens
            if self.config.share_embeddings:
                decoder_identifier_embeds = encoder_outputs.identifier_embeds
            encoder_graph_position_embeds = encoder_outputs.graph_position_embeds
            encoder_graph_position_features = encoder_outputs.graph_position_features
            encoder_orthonormal_features = encoder_outputs.orthonormal_features

        # decoder
        decoder_outputs: GraphsGPTDecoderOutputWithPast = self.decoder(
            decoder_input_ids,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            fingerprint_tokens=fingerprint_tokens,  # encoder output
            attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            identifier_embeds=decoder_identifier_embeds,  # maybe encoder output
            graph_position_embeds=decoder_graph_position_embeds,
            graph_position_features=decoder_graph_position_features,
            orthonormal_features=decoder_orthonormal_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_query_key_values=past_query_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = decoder_outputs.hidden_states
        inputs_embeds = decoder_outputs.inputs_embeds
        identifier_embeds = decoder_outputs.identifier_embeds
        decoder_graph_position_embeds = decoder_outputs.graph_position_embeds
        decoder_graph_position_features = decoder_outputs.graph_position_features
        decoder_orthonormal_features = decoder_outputs.orthonormal_features
        next_cache = decoder_outputs.past_query_key_values
        all_hidden_states = decoder_outputs.all_hidden_states
        all_self_attns = decoder_outputs.all_attentions

        return GraphsGPTModelOutputWithPast(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            encoder_graph_position_embeds=encoder_graph_position_embeds,
            encoder_graph_position_features=encoder_graph_position_features,
            encoder_orthonormal_features=encoder_orthonormal_features,
            decoder_graph_position_embeds=decoder_graph_position_embeds,
            decoder_graph_position_features=decoder_graph_position_features,
            decoder_orthonormal_features=decoder_orthonormal_features,
            fingerprint_tokens=fingerprint_tokens,
            num_fingerprint_tokens=num_fingerprint_tokens,
            all_hidden_states=all_hidden_states,
            all_attentions=all_self_attns,
            past_query_key_values=next_cache,
        )


class GraphsGPTForCausalLM(GraphsGPTPreTrainedModel):
    def __init__(self, config: GraphsGPTConfig):
        super().__init__(config)
        self.model = GraphsGPTModel(config)

        # next bond token prediction head
        self.lm_head = nn.Linear(config.hidden_size, config.bond_vocab_size + 1, bias=False)  # EOS token at position 0

        # first atom token prediction head
        self.n_head = nn.Linear(config.hidden_size, config.atom_vocab_size, bias=False)

        # connection prediction heads
        self.c_head_begin = nn.Linear(config.hidden_size, config.position_feature_size, bias=True)
        self.c_head_end = nn.Linear(config.hidden_size, config.position_feature_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_decoder_graph_position_features(self):
        return self.model.decoder.embed_graph_position.learnable_orthonormal_features.weight

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            # ↓↓↓↓ for decoder layers ↓↓↓↓
            fingerprint_tokens: Optional[torch.Tensor] = None,

            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            past_query_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> GraphsGPTCausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # get outputs
        outputs: GraphsGPTModelOutputWithPast = self.model(
            input_ids,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            attention_mask=attention_mask,  # padding mask, (batch_size, seq_len) or None
            fingerprint_tokens=fingerprint_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_query_key_values=past_query_key_values,
            use_cache=use_cache,
        )

        hidden_states = outputs.hidden_states
        num_fingerprint_tokens = outputs.num_fingerprint_tokens
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1) - 1 - num_fingerprint_tokens  # skip the BOS token and fingerprint tokens
        device = hidden_states.device

        """Next Edge Token Prediction Loss"""
        # get masks
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        bond_mask = attention_mask * ~identifier_ids  # exclude node tokens and padding tokens
        bond_mask_with_bos = torch.cat((
            torch.ones((batch_size, 1), dtype=torch.bool, device=device),
            bond_mask
        ), dim=1)
        bond_mask_add_eos = torch.cat((
            bond_mask,
            torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        ), dim=1)

        # get predictions
        hidden_states_seq = hidden_states[:, num_fingerprint_tokens:, :]  # skip fingerprint tokens, (batch_size, seq_len - num_fingerprint_tokens, hidden_size)
        hidden_states_bond_with_bos = hidden_states_seq[bond_mask_with_bos]  # select BOS and bond tokens, (bond_num + 1, hidden_size)
        logits_lm_bond_with_bos = self.lm_head(hidden_states_bond_with_bos)  # (bond_num + 1, bond_vocab_size + 1)

        # get corrected ids (labels)
        # input_ids: PAD 0, BOS 1, atoms 2-119, bonds 120-?
        # target_input_ids_bond: PAD & BOS & atoms <= 0, bonds 1-?
        target_input_ids_bond = input_ids - self.config.atom_vocab_size - 1
        target_input_ids_bond_add_eos = torch.cat((
            target_input_ids_bond,
            torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
        ), dim=1)
        labels_lm_bond_with_bos = target_input_ids_bond_add_eos[bond_mask_add_eos]  # select bond tokens with an extra 0 (EOS) added to the end of each sequence

        # get loss
        loss = CrossEntropyLoss()(logits_lm_bond_with_bos, labels_lm_bond_with_bos)
        acc = classification_accuracy(logits_lm_bond_with_bos, labels_lm_bond_with_bos)

        """First Node Token Prediction Loss"""
        # get predictions
        hidden_states_bos = hidden_states[:, num_fingerprint_tokens, :]
        logits_n_bos = self.n_head(hidden_states_bos)  # logits of bos tokens

        # get corrected ids (labels)
        # input_ids: PAD 0, BOS 1, atoms 2-119, bonds 120-?
        # labels_node: atoms 0-117
        labels_n_bos = input_ids[:, 0] - 2  # classes of the first atom tokens

        # get loss
        loss_node = CrossEntropyLoss()(logits_n_bos, labels_n_bos) * self.config.node_loss_weight
        acc_node = classification_accuracy(logits_n_bos, labels_n_bos)

        """Connection Prediction Loss"""
        # get masks
        bond_mask_with_bos_without_last = turn_last_true_mask_to_false(bond_mask_with_bos)  # predict for next the bond, (batch_size, seq_len)

        # get normalized prediction features
        hidden_states_bond_with_bos_without_last = hidden_states_seq[bond_mask_with_bos_without_last]  # (bond_num, hidden_size)
        predict_features_c_begin = self.c_head_begin(hidden_states_bond_with_bos_without_last)  # (bond_num, position_feature_size)
        predict_features_c_end = self.c_head_end(hidden_states_bond_with_bos_without_last)
        norm_predict_features_c_begin = F.normalize(predict_features_c_begin, dim=1)
        norm_predict_features_c_end = F.normalize(predict_features_c_end, dim=1)

        # get normalized target features
        # the former "max_atom_num" orthonormal_features in decoder for convergence acceleration
        decoder_orthonormal_features = outputs.decoder_orthonormal_features[0, :, :]  # (max_atom_num, position_feature_size)
        # decoder_orthonormal_features = self.get_decoder_graph_position_features()  # (position_feature_size, position_feature_size)
        norm_decoder_orthonormal_features = F.normalize(decoder_orthonormal_features, dim=1)

        # get similarities & labels for acc
        sim_c_begin = norm_predict_features_c_begin @ norm_decoder_orthonormal_features.t()  # (bond_num, max_atom_num)
        sim_c_end = norm_predict_features_c_end @ norm_decoder_orthonormal_features.t()
        labels_c_begin = graph_position_ids_1[bond_mask]  # (bond_num)
        labels_c_end = graph_position_ids_2[bond_mask]

        # get loss and acc
        if self.config.connection_loss_type == "classification":
            loss_connection_begin = CrossEntropyLoss()(sim_c_begin, labels_c_begin) * self.config.connection_loss_weight
            loss_connection_end = CrossEntropyLoss()(sim_c_end, labels_c_end) * self.config.connection_loss_weight
            loss_connection = loss_connection_begin + loss_connection_end
            acc_connection_begin = classification_accuracy(sim_c_begin, labels_c_begin)
            acc_connection_end = classification_accuracy(sim_c_end, labels_c_end)

        elif self.config.connection_loss_type == "contrastive":
            # similarities for positive loss
            bond_indices = torch.arange(0, labels_c_begin.shape[0], device=device)  # (bond_num)
            sim_c_pos = torch.cat((
                sim_c_begin[bond_indices, labels_c_begin],
                sim_c_end[bond_indices, labels_c_end]
            ), dim=0)

            # similarities for negative loss
            neg_pair_mask = ~torch.eye(norm_decoder_orthonormal_features.shape[0], dtype=torch.bool, device=device)
            sim_c_neg = torch.abs(
                (norm_decoder_orthonormal_features @ norm_decoder_orthonormal_features.t())[neg_pair_mask]
            )

            # summary
            loss_connection_pos = (1 - torch.mean(sim_c_pos)) * self.config.connection_loss_weight
            loss_connection_neg = torch.mean(sim_c_neg) * self.config.connection_loss_weight
            loss_connection = loss_connection_pos + loss_connection_neg
            acc_connection_begin = classification_accuracy(sim_c_begin, labels_c_begin)
            acc_connection_end = classification_accuracy(sim_c_end, labels_c_end)

        else:
            raise NotImplementedError

        return GraphsGPTCausalLMOutputWithPast(
            loss=loss,
            loss_node=loss_node,
            loss_connection=loss_connection,
            acc=acc,
            acc_node=acc_node,
            acc_connection_begin=acc_connection_begin,
            acc_connection_end=acc_connection_end,
            non_padding_token_num=torch.sum(attention_mask).item(),
            bond_num=torch.sum(bond_mask).item(),
            fingerprint_tokens=outputs.fingerprint_tokens,
            past_query_key_values=outputs.past_query_key_values,
            all_hidden_states=outputs.all_hidden_states,
            all_attentions=outputs.all_attentions,
        )

    def encode_to_fingerprints(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            num_fingerprint_tokens: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,
            **kwargs,
    ) -> torch.FloatTensor:
        encoder_outputs: GraphsGPTEncoderOutput = self.model.encoder(
            input_ids,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            num_fingerprint_tokens=num_fingerprint_tokens,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            **kwargs,
        )
        return encoder_outputs.fingerprint_tokens

    def _update_generated_results(
            self,
            generated_results,
            batch_id,
            input_ids,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            fix_aromatic_bond=True,
            inverse_bond_dict=None,
            bond_dict=None,
            **kwargs
    ):
        if fix_aromatic_bond:
            input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids = fix_dissociative_aromatic_bond(
                input_ids,
                graph_position_ids_1,
                graph_position_ids_2,
                identifier_ids,
                inverse_bond_dict,
                bond_dict,
            )

        generated_results[batch_id] = {
            "input_ids": input_ids,  # (this_seq_len)
            "graph_position_ids_1": graph_position_ids_1,  # (this_seq_len)
            "graph_position_ids_2": graph_position_ids_2,  # (this_seq_len)
            "identifier_ids": identifier_ids  # (this_seq_len)
        }
        return generated_results

    def _top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = float('-inf')
        return out

    @no_grad()
    def generate_from_fingerprints(
            self,
            fingerprint_tokens: torch.FloatTensor,
            bond_dict: dict,

            input_ids_list: List[torch.LongTensor] = None,
            graph_position_ids_1_list: List[torch.LongTensor] = None,
            graph_position_ids_2_list: List[torch.LongTensor] = None,
            identifier_ids_list: List[torch.BoolTensor] = None,

            strict_generation: bool = True,  # if "False", will automatically fix the generation when the prediction is invalid
            do_sample: bool = False,  # whether to perform probabilistic sampling for all bond predictions
            topk: int = None,  # (available when "do_sample=True") samples all logits with probability if None
            temperature: float = 1.0,  # (available when "do_sample=True")
            batch_size: int = 1024,  # (available when the model is decoder-only, e.g. "num_fingerprints=0")

            max_atoms: int = None,
            similarity_threshold: float = 0.5,
            check_first_node: bool = True,
            check_atom_valence: bool = False,  # atoms with special electric charges may get mis-killed by this option
            fix_aromatic_bond: bool = False,  # whether to fix the dissociative aromatic bonds by replacing them with single bonds

            use_cache: Optional[bool] = False,
            save_failed: Optional[bool] = False,
            show_progress: Optional[bool] = True,  # show progress bar
            verbose: Optional[bool] = False,  # show failure information
            **kwargs
    ):
        # validity check
        if use_cache:
            # TODO: The cache is not available now.
            # TODO: Fix the bug
            use_cache = False
            warnings.warn('Setting "use_cache=False" as it is bugged!')

        if strict_generation and do_sample:
            warnings.warn(f'Strict generation with sampling can lead to significant generation failures! '
                          f'To enable more flexible generation, please set "strict_generation=False"!')

        # basic configs
        if max_atoms is None:
            max_atoms = self.config.position_feature_size
        else:
            assert max_atoms <= self.config.position_feature_size

        if fingerprint_tokens is not None:
            batch_size = fingerprint_tokens.shape[0]
            num_fingerprint_tokens = fingerprint_tokens.shape[1]
        else:
            num_fingerprint_tokens = 0

        device = self.model.device

        # initialize dict
        inverse_bond_dict = reverse_dict(bond_dict, aggregate_same_results=False)

        # initialize finished molecule sequences
        generated_results = [None for i in range(batch_size)]
        batch_ids_list = [i for i in range(batch_size)]

        # initialize the padding tool
        padding_tool = tensor_stack_padding_collater(0, padding_position="right", return_padding_mask=True)

        # initialize inputs
        if (
                input_ids_list is None and
                graph_position_ids_1_list is None and
                graph_position_ids_2_list is None and
                identifier_ids_list is None
        ):
            input_ids_list = [torch.ones((1,), dtype=torch.int64, device=device) for _ in range(batch_size)]  # initialized as BOS tokens
            graph_position_ids_1_list = [torch.empty((0,), dtype=torch.int64, device=device) for _ in range(batch_size)]
            graph_position_ids_2_list = [torch.empty((0,), dtype=torch.int64, device=device) for _ in range(batch_size)]
            identifier_ids_list = [torch.empty((0,), dtype=torch.bool, device=device) for _ in range(batch_size)]

        elif (
                input_ids_list is not None and
                graph_position_ids_1_list is not None and
                graph_position_ids_2_list is not None and
                identifier_ids_list is not None
        ):
            input_ids_list = input_ids_list
            graph_position_ids_1_list = graph_position_ids_1_list
            graph_position_ids_2_list = graph_position_ids_2_list
            identifier_ids_list = identifier_ids_list

        else:
            raise ValueError(f"The initializing inputs must be all \"Nones\" or no \"None\"!")

        # initialize cached inputs embeds
        past_query_key_values = None

        # start auto-regressive generation
        self.model.eval()
        while batch_size > 0:
            # pad inputs
            input_ids, attention_mask = padding_tool(input_ids_list)
            graph_position_ids_1, _ = padding_tool(graph_position_ids_1_list)
            graph_position_ids_2, _ = padding_tool(graph_position_ids_2_list)
            identifier_ids, _ = padding_tool(identifier_ids_list)
            seq_len = input_ids.shape[1]

            # forward decoder
            outputs: GraphsGPTDecoderOutputWithPast = self.model.decoder(
                input_ids,
                graph_position_ids_1,
                graph_position_ids_2,
                identifier_ids,
                fingerprint_tokens=fingerprint_tokens,
                attention_mask=attention_mask,
                inputs_embeds=None,
                identifier_embeds=None,
                graph_position_embeds=None,  # graph_position_embeds
                graph_position_features=None,
                output_attentions=None,
                output_hidden_states=None,
                past_query_key_values=past_query_key_values if seq_len > 1 else None,
                use_cache=use_cache,
            )

            # get outputs
            hidden_states = outputs.hidden_states
            orthonormal_features = outputs.orthonormal_features
            past_query_key_values = outputs.past_query_key_values

            # get the indices of tokens for prediction
            batch_positions = torch.arange(0, batch_size, step=1, device=device)

            if seq_len == 1:  # generate from BOS tokens
                last_edge_positions = torch.full((batch_size,), num_fingerprint_tokens, dtype=torch.int64, device=device)  # position of the BOS token
            else:  # generate from incomplete sequences
                bond_mask = ~identifier_ids & attention_mask[:, 1:]
                last_edge_positions = last_true_position(bond_mask).squeeze(1) + num_fingerprint_tokens + 1  # position of the last edge token

            # select hidden states for prediction
            bos_hidden_states = hidden_states[:, num_fingerprint_tokens, :] if seq_len == 1 else None
            bond_hidden_states = hidden_states[batch_positions, last_edge_positions]

            # get logits & features
            logits_lm = self.lm_head(bond_hidden_states)  # bond type
            logits_n = self.n_head(bos_hidden_states) if seq_len == 1 else None  # node type
            features_begin = self.c_head_begin(bond_hidden_states)  # connection begin
            features_end = self.c_head_end(bond_hidden_states)  # connection end

            # create bond candidates in case the generation failed
            # the candidate list only works when "strict_generation=False"
            _, candidate_preds_lm = torch.sort(logits_lm, descending=True, dim=1)

            # get predictions of next tokens
            if do_sample:
                logits_lm /= temperature
                if topk is not None:
                    logits_lm = self._top_k_logits(logits_lm, topk)
                probs_lm = F.softmax(logits_lm, dim=1)
                preds_lm = torch.multinomial(probs_lm, num_samples=1).squeeze(1)
            else:
                preds_lm = torch.argmax(logits_lm, dim=1)

            preds_n = torch.argmax(logits_n, dim=1) if seq_len == 1 else None

            # update generated results
            next_batch_ids_list = []
            next_input_ids_list = []
            next_graph_position_ids_1_list = []
            next_graph_position_ids_2_list = []
            next_identifier_ids_list = []

            # indicator - end of prediction
            pred_is_end = torch.ones((batch_size,), dtype=torch.bool, device=device)  # prediction is EOS, (batch_size)

            # iterate by sample
            for i in tqdm(range(batch_size)) if show_progress else range(batch_size):
                # basic information
                batch_id = batch_ids_list[i]
                this_padding_mask = attention_mask[i, :]  # (seq_len)
                this_atom_mask = (this_padding_mask[1:] & identifier_ids[i, :])  # (seq_len - 1)
                this_bond_mask = (this_padding_mask[1:] & ~identifier_ids[i, :])  # (seq_len - 1)
                this_seq_len = torch.sum(this_padding_mask).item()
                atom_num = torch.sum(this_atom_mask).item()

                # construct the candidate prediction list (move the "preds_lm" to the top of "candidate_preds_lm")
                this_true_pred_lm = preds_lm[i].item()
                this_candidate_pred_lm = candidate_preds_lm[i].tolist()
                this_candidate_pred_lm.remove(this_true_pred_lm)
                this_pred_lm_list = [this_true_pred_lm] + this_candidate_pred_lm

                # iterate over the candidate predictions
                for this_pred_lm in this_pred_lm_list:
                    this_pred_lm = torch.tensor(this_pred_lm, dtype=torch.int64, device=device)
                    pred_is_end[i] = (this_pred_lm == 0)

                    ### meet the EOS token ###
                    if pred_is_end[i]:
                        if this_seq_len > 1:  # generation finished, add to generated results
                            generated_results = self._update_generated_results(
                                generated_results,
                                batch_id,
                                input_ids_list[i][1:],
                                graph_position_ids_1_list[i],
                                graph_position_ids_2_list[i],
                                identifier_ids_list[i],
                                fix_aromatic_bond=fix_aromatic_bond,
                                inverse_bond_dict=inverse_bond_dict,
                                bond_dict=bond_dict,
                            )
                            break
                        else:
                            if strict_generation:  # generation failed
                                if verbose:
                                    print(f"Generation of sample {batch_id} failed. "
                                          f"Reason: get EOS prediction for the first bond.")
                                break
                            else:  # try the next bond type
                                if verbose:
                                    print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (Got EOS For First Bond)")
                                continue

                    else:
                        # prepare the next input
                        # dict: atoms 1-118, bonds 119-?
                        # input_ids: PAD 0, BOS 1, atoms 2-119, bonds 120-?
                        next_bond_id_in_dict = this_pred_lm.unsqueeze(0) + self.config.atom_vocab_size  # id in the bond_dict

                        ### predict from BOS ###
                        if seq_len == 1:
                            next_atom_id_1_in_dict, next_atom_id_2_in_dict = get_atom_ids_from_bond_id(inverse_bond_dict, next_bond_id_in_dict)

                            if check_first_node:
                                predicted_first_atom_id_in_dict = preds_n[i] + 1  # "+1" for getting the correct atomic num

                                # check the availability of predicted atom id
                                if predicted_first_atom_id_in_dict != next_atom_id_1_in_dict and predicted_first_atom_id_in_dict != next_atom_id_2_in_dict:
                                    if strict_generation:  # generation failed
                                        pred_is_end[i] = True
                                        if verbose:
                                            print(f"Generation of sample {batch_id} failed. "
                                                  f"Reason: the predicted first atom does not match the predicted first bond. "
                                                  f"(Required predicted atom id in [{next_atom_id_1_in_dict.item()},{next_atom_id_2_in_dict.item()}], got {predicted_first_atom_id_in_dict.item()})")
                                        break
                                    else:  # try the next bond type
                                        if verbose:
                                            print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (First Node Not Match)")
                                        continue

                                # correct the first atom id
                                elif predicted_first_atom_id_in_dict != next_atom_id_1_in_dict:
                                    next_atom_id_1_in_dict, next_atom_id_2_in_dict = (next_atom_id_2_in_dict, next_atom_id_1_in_dict)

                            next_input_ids = torch.cat((input_ids_list[i], next_atom_id_1_in_dict + 1, next_bond_id_in_dict + 1, next_atom_id_2_in_dict + 1))  # shape(4), "+1" for skip the padding and BOS embedding position
                            next_graph_position_ids_1 = torch.tensor((0, 0, 1), dtype=torch.int64, device=device)  # shape(3)
                            next_graph_position_ids_2 = torch.tensor((0, 1, 1), dtype=torch.int64, device=device)  # shape(3)
                            next_identifier_ids = torch.tensor((True, False, True), dtype=torch.bool, device=device)  # shape(3)

                        ### predict from existing sequences ###
                        else:
                            # get similarities between predicted features and input features
                            atom_ids_in_dict = input_ids[i, 1:][this_atom_mask] - 1  # shape(atom_num), "-1" for skipping padding and BOS tokens

                            norm_orthonormal_features = F.normalize(orthonormal_features[i, :atom_num, :].squeeze(0), dim=1)  # (atom_num, position_feature_size)
                            norm_predict_features_begin = F.normalize(features_begin[i].unsqueeze(0), dim=1)  # (1, position_feature_size)
                            norm_predict_features_end = F.normalize(features_end[i].unsqueeze(0), dim=1)  # (1, position_feature_size)

                            similarities_begin = torch.mm(norm_orthonormal_features, norm_predict_features_begin.transpose(0, 1))  # (atom_num, 1)
                            similarities_end = torch.mm(norm_orthonormal_features, norm_predict_features_end.transpose(0, 1))  # (atom_num, 1)
                            max_similarity_begin, most_similar_position_begin = torch.max(similarities_begin, dim=0)
                            max_similarity_end, most_similar_position_end = torch.max(similarities_end, dim=0)

                            most_similar_atom_id_in_dict_begin = atom_ids_in_dict[most_similar_position_begin]
                            most_similar_atom_id_in_dict_end = atom_ids_in_dict[most_similar_position_end]

                            # check the validity of atom valences
                            if check_atom_valence:
                                # begin atom
                                if most_similar_atom_id_in_dict_begin.item() in VALENCE_LIMIT:
                                    begin_total_valence, begin_valence_offset = get_valence(input_ids[i, 1:], graph_position_ids_1[i], graph_position_ids_2[i], most_similar_position_begin.item(), inverse_bond_dict, this_bond_mask)
                                    max_valence = VALENCE_LIMIT[most_similar_atom_id_in_dict_begin.item()]
                                    # begin valence is full
                                    if begin_total_valence >= max_valence + begin_valence_offset:
                                        valence_fixed = False
                                        if not strict_generation:  # try the next connection
                                            # iteratively search the positions with maximum similarities until the valence is valid
                                            for _ in range(min(atom_num - 1, 10)):  # search for the new connection for at most 10 times
                                                if verbose:
                                                    print(f"Avoiding \"begin_connection={most_similar_position_begin.item()}\" for sample {batch_id} to prevent generation failure. (Begin Valence Limit)")
                                                similarities_begin[most_similar_position_begin] = -100
                                                max_similarity_begin, most_similar_position_begin = torch.max(similarities_begin, dim=0)
                                                most_similar_atom_id_in_dict_begin = atom_ids_in_dict[most_similar_position_begin]
                                                begin_total_valence, begin_valence_offset = get_valence(input_ids[i, 1:], graph_position_ids_1[i], graph_position_ids_2[i], most_similar_position_begin.item(), inverse_bond_dict, this_bond_mask)
                                                if begin_total_valence < max_valence + begin_valence_offset:
                                                    valence_fixed = True
                                                    break
                                        if not valence_fixed:  # generation failed
                                            pred_is_end[i] = True
                                            if verbose:
                                                print(f"Generation of sample {batch_id} failed. "
                                                      f"Reason: the valence ({begin_total_valence}) of connected begin atom (atomic_num={most_similar_atom_id_in_dict_begin.item()}) has been full ({max_valence} ± {begin_valence_offset}).")
                                            if save_failed:
                                                generated_results = self._update_generated_results(
                                                    generated_results,
                                                    batch_id,
                                                    input_ids_list[i][1:],
                                                    graph_position_ids_1_list[i],
                                                    graph_position_ids_2_list[i],
                                                    identifier_ids_list[i],
                                                    fix_aromatic_bond=fix_aromatic_bond,
                                                    inverse_bond_dict=inverse_bond_dict,
                                                    bond_dict=bond_dict,
                                                )
                                            break

                                    new_valence = inverse_bond_dict[next_bond_id_in_dict.item()][2]
                                    new_valence_offset = 0.5 if inverse_bond_dict[next_bond_id_in_dict.item()][2] == 1.5 else 0.0
                                    total_valence_offset = begin_valence_offset + new_valence_offset
                                    # begin valence exceeds the limit
                                    if begin_total_valence + new_valence > max_valence + total_valence_offset:
                                        if strict_generation:  # generation failed
                                            pred_is_end[i] = True
                                            if verbose:
                                                print(f"Generation of sample {batch_id} failed. "
                                                      f"Reason: the valence ({begin_total_valence} + {new_valence}) of connected begin atom (atomic_num={most_similar_atom_id_in_dict_begin.item()}) exceeds the limit ({max_valence} ± {total_valence_offset}).")
                                            if save_failed:
                                                generated_results = self._update_generated_results(
                                                    generated_results,
                                                    batch_id,
                                                    input_ids_list[i][1:],
                                                    graph_position_ids_1_list[i],
                                                    graph_position_ids_2_list[i],
                                                    identifier_ids_list[i],
                                                    fix_aromatic_bond=fix_aromatic_bond,
                                                    inverse_bond_dict=inverse_bond_dict,
                                                    bond_dict=bond_dict,
                                                )
                                            break
                                        else:  # try the next bond type
                                            if verbose:
                                                print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (Begin Valence Limit)")
                                            continue

                                # end atom
                                if max_similarity_end > similarity_threshold and most_similar_atom_id_in_dict_end.item() in VALENCE_LIMIT:
                                    end_total_valence, end_valence_offset = get_valence(input_ids[i, 1:], graph_position_ids_1[i], graph_position_ids_2[i], most_similar_position_end.item(), inverse_bond_dict, this_bond_mask)
                                    max_valence = VALENCE_LIMIT[most_similar_atom_id_in_dict_end.item()]
                                    # end valence is full
                                    if end_total_valence >= max_valence + end_valence_offset:
                                        valence_fixed = False
                                        if not strict_generation:  # try the next connection
                                            # iteratively search the positions with maximum similarities until the valence is valid
                                            for _ in range(min(atom_num - 1, 10)):  # search for the new connection for at most 10 times
                                                if verbose:
                                                    print(f"Avoiding \"end_connection={most_similar_position_end.item()}\" for sample {batch_id} to prevent generation failure. (End Valence Limit)")
                                                similarities_end[most_similar_position_end] = -100
                                                max_similarity_end, most_similar_position_end = torch.max(similarities_end, dim=0)
                                                most_similar_atom_id_in_dict_end = atom_ids_in_dict[most_similar_position_end]
                                                end_total_valence, end_valence_offset = get_valence(input_ids[i, 1:], graph_position_ids_1[i], graph_position_ids_2[i], most_similar_position_end.item(), inverse_bond_dict, this_bond_mask)
                                                if end_total_valence < max_valence + end_valence_offset:
                                                    valence_fixed = True
                                                    break
                                        if not valence_fixed:  # generation failed
                                            pred_is_end[i] = True
                                            if verbose:
                                                print(f"Generation of sample {batch_id} failed. "
                                                      f"Reason: the valence ({end_total_valence}) of connected end atom (atomic_num={most_similar_atom_id_in_dict_end.item()}) has been full ({max_valence} ± {end_valence_offset}).")
                                            if save_failed:
                                                generated_results = self._update_generated_results(
                                                    generated_results,
                                                    batch_id,
                                                    input_ids_list[i][1:],
                                                    graph_position_ids_1_list[i],
                                                    graph_position_ids_2_list[i],
                                                    identifier_ids_list[i],
                                                    fix_aromatic_bond=fix_aromatic_bond,
                                                    inverse_bond_dict=inverse_bond_dict,
                                                    bond_dict=bond_dict,
                                                )
                                            break

                                    new_valence = inverse_bond_dict[next_bond_id_in_dict.item()][2]
                                    new_valence_offset = 0.5 if inverse_bond_dict[next_bond_id_in_dict.item()][2] == 1.5 else 0.0
                                    total_valence_offset = end_valence_offset + new_valence_offset
                                    # end valence exceeds the limit
                                    if end_total_valence + new_valence > max_valence + total_valence_offset:
                                        if strict_generation:  # generation failed
                                            pred_is_end[i] = True
                                            if verbose:
                                                print(f"Generation of sample {batch_id} failed. "
                                                      f"Reason: the valence ({end_total_valence} + {new_valence}) of connected end atom (atomic_num={most_similar_atom_id_in_dict_end.item()}) exceeds the limit ({max_valence} ± {total_valence_offset}).")
                                            if save_failed:
                                                generated_results = self._update_generated_results(
                                                    generated_results,
                                                    batch_id,
                                                    input_ids_list[i][1:],
                                                    graph_position_ids_1_list[i],
                                                    graph_position_ids_2_list[i],
                                                    identifier_ids_list[i],
                                                    fix_aromatic_bond=fix_aromatic_bond,
                                                    inverse_bond_dict=inverse_bond_dict,
                                                    bond_dict=bond_dict,
                                                )
                                            break
                                        else:  # try the next bond type
                                            if verbose:
                                                print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (End Valence Limit)")
                                            continue

                            ### the bond connects to two existed atoms ###
                            ### check the validity of the connected atoms ###
                            connect_to_new_atom = False
                            if max_similarity_end > similarity_threshold:
                                # validity check failed, the bond connects to the same atom
                                if most_similar_position_begin == most_similar_position_end:
                                    if strict_generation:  # generation failed
                                        pred_is_end[i] = True
                                        if verbose:
                                            print(f"Generation of sample {batch_id} failed. "
                                                  f"Reason: the bond connects to the same atom (position id {most_similar_position_begin.item()})")
                                        if save_failed:
                                            generated_results = self._update_generated_results(
                                                generated_results,
                                                batch_id,
                                                input_ids_list[i][1:],
                                                graph_position_ids_1_list[i],
                                                graph_position_ids_2_list[i],
                                                identifier_ids_list[i],
                                                fix_aromatic_bond=fix_aromatic_bond,
                                                inverse_bond_dict=inverse_bond_dict,
                                                bond_dict=bond_dict,
                                            )
                                        break
                                    else:  # change the connection if not in strict mode
                                        if verbose:
                                            print(f"Avoiding connecting to the existing atoms for sample {batch_id} to prevent generation failure. (Connect to Same Atom)")
                                        connect_to_new_atom = True  # we don't continue here

                                # validity check failed, the connection has been existed in the graph
                                elif check_bond_in_graph(graph_position_ids_1[i], graph_position_ids_2[i], most_similar_position_begin.item(), most_similar_position_end.item()):
                                    if strict_generation:  # generation failed
                                        pred_is_end[i] = True
                                        if verbose:
                                            print(f"Generation of sample {batch_id} failed. "
                                                  f"Reason: connection {most_similar_position_begin.item(), most_similar_position_end.item()} has already existed in the graph. ")
                                        if save_failed:
                                            generated_results = self._update_generated_results(
                                                generated_results,
                                                batch_id,
                                                input_ids_list[i][1:],
                                                graph_position_ids_1_list[i],
                                                graph_position_ids_2_list[i],
                                                identifier_ids_list[i],
                                                fix_aromatic_bond=fix_aromatic_bond,
                                                inverse_bond_dict=inverse_bond_dict,
                                                bond_dict=bond_dict,
                                            )
                                        break
                                    else:  # change the connection if not in strict mode
                                        if verbose:
                                            print(f"Avoiding connecting to the existing atoms for sample {batch_id} to prevent generation failure. (Bond Existed)")
                                        connect_to_new_atom = True  # we don't continue here

                                # validity check failed, the connection must perfectly match the predicted bond type
                                elif not check_bond_connectivity_both_sides(inverse_bond_dict, next_bond_id_in_dict.item(), most_similar_atom_id_in_dict_begin.item(), most_similar_atom_id_in_dict_end.item()):
                                    if strict_generation:  # generation failed
                                        pred_is_end[i] = True
                                        if verbose:
                                            print(f"Generation of sample {batch_id} failed. "
                                                  f"Reason: atoms connected by the generated bond is invalid. "
                                                  f"(Required connected atomic num: "
                                                  f"{inverse_bond_dict[next_bond_id_in_dict.item()][0]} and {inverse_bond_dict[next_bond_id_in_dict.item()][1]}, got "
                                                  f"{most_similar_atom_id_in_dict_begin.item()} and {most_similar_atom_id_in_dict_end.item()})")
                                        if save_failed:
                                            generated_results = self._update_generated_results(
                                                generated_results,
                                                batch_id,
                                                input_ids_list[i][1:],
                                                graph_position_ids_1_list[i],
                                                graph_position_ids_2_list[i],
                                                identifier_ids_list[i],
                                                fix_aromatic_bond=fix_aromatic_bond,
                                                inverse_bond_dict=inverse_bond_dict,
                                                bond_dict=bond_dict,
                                            )
                                        break
                                    else:  # try the next bond type
                                        if verbose:
                                            print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (Connected Existing Atoms Invalid)")
                                        continue

                                # validity check passed, add to inputs
                                else:
                                    next_input_ids = torch.cat((input_ids_list[i], next_bond_id_in_dict + 1), dim=0)  # (this_seq_len + 1)
                                    next_graph_position_ids_1 = torch.cat((graph_position_ids_1_list[i], most_similar_position_begin), dim=0)  # (this_seq_len + 1)
                                    next_graph_position_ids_2 = torch.cat((graph_position_ids_2_list[i], most_similar_position_end), dim=0)  # (this_seq_len + 1)
                                    next_identifier_ids = torch.cat((identifier_ids_list[i], torch.tensor((False,), dtype=torch.bool, device=device)), dim=0)  # (this_seq_len + 1)

                            ### the bond connects to a new atom ###
                            ### check the validity of the new atom ###
                            if max_similarity_end <= similarity_threshold or connect_to_new_atom:
                                # reached atom number limit
                                if atom_num >= max_atoms:
                                    pred_is_end[i] = True
                                    if verbose:
                                        print(f"Generation of sample {batch_id} failed. "
                                              f"Reason: reached the limit of maximum number of atoms ({max_atoms}).")
                                    if save_failed:
                                        generated_results = self._update_generated_results(
                                            generated_results,
                                            batch_id,
                                            input_ids_list[i][1:],
                                            graph_position_ids_1_list[i],
                                            graph_position_ids_2_list[i],
                                            identifier_ids_list[i],
                                            fix_aromatic_bond=fix_aromatic_bond,
                                            inverse_bond_dict=inverse_bond_dict,
                                            bond_dict=bond_dict,
                                        )
                                    break

                                # validity check failed, the bond cannot connect to the begin atom
                                elif not check_bond_connectivity_begin(inverse_bond_dict, next_bond_id_in_dict.item(), most_similar_atom_id_in_dict_begin.item()):
                                    if strict_generation:  # generation failed
                                        pred_is_end[i] = True
                                        if verbose:
                                            print(f"Generation of sample {batch_id} failed. "
                                                  f"Reason: the begin atom connected by the generated bond is invalid. "
                                                  f"(Required connected begin atomic num: "
                                                  f"{inverse_bond_dict[next_bond_id_in_dict.item()][0]} or {inverse_bond_dict[next_bond_id_in_dict.item()][1]}, got "
                                                  f"{most_similar_atom_id_in_dict_begin.item()})")
                                        if save_failed:
                                            generated_results = self._update_generated_results(
                                                generated_results,
                                                batch_id,
                                                input_ids_list[i][1:],
                                                graph_position_ids_1_list[i],
                                                graph_position_ids_2_list[i],
                                                identifier_ids_list[i],
                                                fix_aromatic_bond=fix_aromatic_bond,
                                                inverse_bond_dict=inverse_bond_dict,
                                                bond_dict=bond_dict,
                                            )
                                        break
                                    else:  # try the next bond type
                                        if verbose:
                                            print(f"Avoiding \"pred_lm={this_pred_lm.item()}\" for sample {batch_id} to prevent generation failure. (Connected New Atom Invalid)")
                                        continue

                                # validity check passed, add to inputs
                                else:
                                    next_atom_id_in_dict = get_another_atom_id_from_existing_bond(inverse_bond_dict, next_bond_id_in_dict, most_similar_atom_id_in_dict_begin)
                                    next_atom_position = torch.tensor((atom_num,), dtype=torch.int64, device=device)

                                    next_input_ids = torch.cat((input_ids_list[i], next_bond_id_in_dict + 1, next_atom_id_in_dict + 1), dim=0)  # (this_seq_len + 2)
                                    next_graph_position_ids_1 = torch.cat((graph_position_ids_1_list[i], most_similar_position_begin, next_atom_position), dim=0)  # (this_seq_len + 2)
                                    next_graph_position_ids_2 = torch.cat((graph_position_ids_2_list[i], next_atom_position, next_atom_position), dim=0)  # (this_seq_len + 2)
                                    next_identifier_ids = torch.cat((identifier_ids_list[i], torch.tensor((False, True), dtype=torch.bool, device=device)), dim=0)  # (this_seq_len + 2)

                        # generation validation passed, prepare for the next iteration
                        next_batch_ids_list.append(batch_id)
                        next_input_ids_list.append(next_input_ids)
                        next_graph_position_ids_1_list.append(next_graph_position_ids_1)
                        next_graph_position_ids_2_list.append(next_graph_position_ids_2)
                        next_identifier_ids_list.append(next_identifier_ids)
                        break

            # get inputs for the next generation
            batch_size = len(next_batch_ids_list)

            if batch_size > 0:
                batch_ids_list = next_batch_ids_list
                input_ids_list = next_input_ids_list
                graph_position_ids_1_list = next_graph_position_ids_1_list
                graph_position_ids_2_list = next_graph_position_ids_2_list
                identifier_ids_list = next_identifier_ids_list

                if num_fingerprint_tokens > 0:
                    fingerprint_tokens = fingerprint_tokens[~pred_is_end]

                if use_cache:
                    new_past_query_key_values = ()
                    for layer_id in range(len(past_query_key_values)):
                        this_layer_past_query_key_values = (
                            past_query_key_values[layer_id][0][~pred_is_end],
                            past_query_key_values[layer_id][1][~pred_is_end],
                            past_query_key_values[layer_id][2][~pred_is_end],
                        )
                        new_past_query_key_values += (this_layer_past_query_key_values,)
                    past_query_key_values = new_past_query_key_values

        return generated_results
