import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import logging, ModelOutput
from typing import Optional, List, Tuple

from models.graphsgpt.modeling_graphsgpt import GraphsGPTPreTrainedModel, GraphsGPTDecoder, GraphsGPTForCausalLM, GraphsGPTEncoderLayer, StableEmbedding, RMSNorm, GraphPositionStableEmbedding, _expand_mask, GraphPositionEmbeddingOutput, GraphsGPTDecoderOutputWithPast, GraphsGPTModelOutputWithPast, GraphsGPTCausalLMOutputWithPast
from models.graphsgpt_cond_gen.configuration_graphsgpt_cond_gen import GraphsGPTConditionedConfig
from utils.accuracy import classification_accuracy
from utils.operations.operation_tensor import turn_last_true_mask_to_false

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GraphsGPTConditionedConfig"


@dataclass
class GraphsGPTEncoderConditionedOutput(ModelOutput):
    fingerprint_tokens: torch.FloatTensor = None

    values_properties: torch.FloatTensor = None  # 🔍
    indices_properties: torch.LongTensor = None  # 🔍

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None
    graph_position_embeds: Optional[torch.FloatTensor] = None
    graph_position_features: Optional[torch.FloatTensor] = None
    orthonormal_features: Optional[torch.FloatTensor] = None
    graph_embedding_ids: Optional[torch.LongTensor] = None

    attention_mask: Optional[torch.Tensor] = None


@dataclass
class GraphsGPTModelConditionedOutputWithPast(ModelOutput):
    hidden_states: torch.FloatTensor = None

    inputs_embeds: Optional[torch.FloatTensor] = None
    identifier_embeds: Optional[torch.FloatTensor] = None

    encoder_graph_position_embeds: Optional[torch.FloatTensor] = None
    encoder_graph_position_features: Optional[torch.FloatTensor] = None
    encoder_orthonormal_features: Optional[torch.FloatTensor] = None
    encoder_graph_embedding_ids: Optional[torch.LongTensor] = None

    decoder_graph_position_embeds: Optional[torch.FloatTensor] = None
    decoder_graph_position_features: Optional[torch.FloatTensor] = None
    decoder_orthonormal_features: Optional[torch.FloatTensor] = None
    decoder_graph_embedding_ids: Optional[torch.LongTensor] = None

    attention_mask: Optional[torch.Tensor] = None
    fingerprint_tokens: torch.FloatTensor = None
    values_properties: torch.FloatTensor = None  # 🔍
    indices_properties: torch.LongTensor = None  # 🔍
    num_fingerprint_tokens: int = None

    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_query_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


class ConditionStableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.norm = torch.nn.LayerNorm(embedding_dim)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, values, indices=None) -> torch.Tensor:
        if indices is None:
            # values: shape(batch_size, num_embeddings)
            # return: shape(batch_size, num_embeddings, embedding_dim)
            features = torch.einsum('bn,ne->bne', values, self.weight)
            features = self.norm(features)
        else:
            # values: shape(batch_size, this_num_embeddings)
            # indices: shape(batch_size, this_num_embeddings)
            # return: shape(batch_size, this_num_embeddings, embedding_dim)
            selected_weight = torch.index_select(self.weight, dim=0, index=indices)
            features = torch.einsum('bn,ne->bne', values, selected_weight)
            features = self.norm(features)

        return features


class GraphsGPTEncoderConditioned(nn.Module):
    def __init__(self, config: GraphsGPTConditionedConfig):
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
        self.embed_fingerprint = StableEmbedding(1, config.hidden_size)
        self.embed_fingerprint_position = StableEmbedding(config.num_fingerprints, config.hidden_size)

        # graph position embeddings
        self.embed_graph_position = GraphPositionStableEmbedding(config.position_feature_size, config.hidden_size)

        # 🔍 property embeddings
        self.embed_property = ConditionStableEmbedding(config.num_properties, config.hidden_size)
        self.embed_property_position = StableEmbedding(config.num_properties, config.hidden_size)

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

    # 🔍
    def concat_properties(
            self,
            values_qed: torch.FloatTensor = None,
            values_sa: torch.FloatTensor = None,
            values_logp: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        values_properties = []
        indices_properties = []

        if values_qed is not None:
            values_properties.append(values_qed)
            indices_properties.append(0)
        if values_sa is not None:
            values_properties.append(values_sa)
            indices_properties.append(1)
        if values_logp is not None:
            values_properties.append(values_logp)
            indices_properties.append(2)

        if len(values_properties) == 0:
            return None, None
        else:
            values_properties = torch.cat(values_properties, dim=1)
            indices_properties = torch.tensor(indices_properties, dtype=torch.int64, device=values_properties.device)
            return values_properties, indices_properties

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,

            values_qed: torch.FloatTensor = None,  # 🔍
            values_sa: torch.FloatTensor = None,  # 🔍
            values_logp: torch.FloatTensor = None,  # 🔍

            num_fingerprint_tokens: Optional[int] = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,
            graph_embedding_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> GraphsGPTEncoderConditionedOutput:
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

        # 🔍 initialize properties
        values_properties, indices_properties = self.concat_properties(
            values_qed=values_qed,
            values_sa=values_sa,
            values_logp=values_logp,
        )

        # 🔍 property validity check
        if values_properties is None:
            raise ValueError("You must specify at least one condition!")

        num_properties = values_properties.shape[1]

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

        # 🔍
        property_embeds = self.embed_property(values_properties, indices=indices_properties)  # (batch_size, num_properties, embed_dim)
        property_position_embeds = self.embed_property_position(indices_properties)  # (batch_size, num_properties, embed_dim)

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
                embedding_ids=graph_embedding_ids,
                orthonormal_features=orthonormal_features,
                graph_position_features=graph_position_features,
                return_features=False,
            )
            graph_position_embeds = graph_embedding_outputs.graph_position_embeds  # (batch_size, seq_len, embed_dim)
            graph_position_features = graph_embedding_outputs.graph_position_features  # None
            orthonormal_features = graph_embedding_outputs.orthonormal_features  # None
            graph_embedding_ids = graph_embedding_outputs.embedding_ids

        if identifier_embeds is None:
            identifier_embeds = self.embed_identifier(identifier_ids.clone().int())  # (batch_size, seq_len, embed_dim)

        # add embeds together and get hidden_states
        fingerprint_tokens = fingerprint_embeds + fingerprint_position_embeds  # (batch_size, num_fingerprint_tokens, embed_dim)
        property_tokens = property_embeds + property_position_embeds
        molecule_tokens = inputs_embeds + graph_position_embeds + identifier_embeds  # (batch_size, seq_len, embed_dim)

        # 🔍
        hidden_states = torch.cat((fingerprint_tokens, property_tokens, molecule_tokens), dim=1)  # (batch_size, num_fingerprint_tokens + num_properties + seq_len, embed_dim)

        # get attention masks
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, num_fingerprint_tokens + num_properties + seq_length), dtype=torch.bool, device=device)  # 🔍
        else:  # the attention mask is shaped like (batch_size, mole_seq_len) so we need to extend its dimension
            extra_dim = num_fingerprint_tokens + num_properties + seq_length - attention_mask.shape[1]  # 🔍
            if extra_dim > 0:  # adding extra dimensions to the attention mask
                extra_attention_mask = torch.ones((batch_size, extra_dim), dtype=torch.bool, device=device)
                attention_mask = torch.cat((extra_attention_mask, attention_mask), dim=1)
            else:
                attention_mask = attention_mask

        attention_mask = self._prepare_encoder_attention_mask(
            attention_mask,
            (batch_size, num_fingerprint_tokens + num_properties + seq_length),  # 🔍
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

        return GraphsGPTEncoderConditionedOutput(
            fingerprint_tokens=fingerprint_tokens,
            values_properties=values_properties,  # 🔍
            indices_properties=indices_properties,  # 🔍
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            graph_embedding_ids=graph_embedding_ids,
            attention_mask=attention_mask,
        )


class GraphsGPTConditionedPreTrainedModel(GraphsGPTPreTrainedModel):
    config_class = GraphsGPTConditionedConfig


class GraphsGPTModelConditioned(GraphsGPTConditionedPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GraphsGPTDecoderLayer`]

    Args:
        config: GraphsGPTConditionedConfig
    """

    def __init__(self, config: GraphsGPTConditionedConfig):
        super().__init__(config)
        if config.num_fingerprints > 0:  # encoder-decoder
            self.encoder = GraphsGPTEncoderConditioned(config)  # 🔍
            self.decoder = GraphsGPTDecoder(
                config,
                embed_tokens=None if not config.share_embeddings else self.encoder_condition.embed_tokens,
                embed_identifier=None if not config.share_embeddings else self.encoder_condition.embed_identifier,
                embed_fingerprint_position=None if not config.share_embeddings else self.encoder_condition.embed_fingerprint_position,
                embed_graph_position=None if not config.share_embeddings else self.encoder_condition.embed_graph_position,
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

            scaffold_input_ids: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_1: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_2: torch.LongTensor = None,  # 🔍
            scaffold_identifier_ids: torch.BoolTensor = None,  # 🔍
            scaffold_attention_mask: Optional[torch.Tensor] = None,  # 🔍 padding mask

            values_qed: torch.FloatTensor = None,  # 🔍
            values_sa: torch.FloatTensor = None,  # 🔍
            values_logp: torch.FloatTensor = None,  # 🔍

            # ↓↓↓↓ for encoder layers ↓↓↓↓
            encoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_identifier_embeds: Optional[torch.FloatTensor] = None,
            encoder_graph_position_embeds: Optional[torch.FloatTensor] = None,
            encoder_graph_position_features: Optional[torch.FloatTensor] = None,
            encoder_orthonormal_features: Optional[torch.FloatTensor] = None,
            encoder_graph_embedding_ids: Optional[torch.LongTensor] = None,

            # ↓↓↓↓ for decoder layers ↓↓↓↓
            fingerprint_tokens: Optional[torch.Tensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_identifier_embeds: Optional[torch.FloatTensor] = None,
            decoder_graph_position_embeds: Optional[torch.FloatTensor] = None,
            decoder_graph_position_features: Optional[torch.FloatTensor] = None,
            decoder_orthonormal_features: Optional[torch.FloatTensor] = None,
            decoder_graph_embedding_ids: Optional[torch.LongTensor] = None,

            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            past_query_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            **kwargs,
    ) -> GraphsGPTModelConditionedOutputWithPast:
        """forward function for end-to-end model training"""
        decoder_input_ids = torch.cat((
            torch.ones((input_ids.shape[0], 1), dtype=torch.int64, device=input_ids.device),
            input_ids
        ), dim=1)  # add BOS token on the left side

        if fingerprint_tokens is None:
            num_fingerprint_tokens = self.config.num_fingerprints
        else:
            num_fingerprint_tokens = fingerprint_tokens.shape[1]

        # 🔍 encoder
        if num_fingerprint_tokens <= 0:
            fingerprint_tokens = None
            values_properties = None
            indices_properties = None
        else:
            encoder_outputs: GraphsGPTEncoderConditionedOutput = self.encoder(
                scaffold_input_ids,
                scaffold_graph_position_ids_1,
                scaffold_graph_position_ids_2,
                scaffold_identifier_ids,
                values_qed,
                values_sa,
                values_logp,
                num_fingerprint_tokens=num_fingerprint_tokens,
                attention_mask=scaffold_attention_mask,
                inputs_embeds=encoder_inputs_embeds,
                identifier_embeds=encoder_identifier_embeds,
                graph_position_embeds=encoder_graph_position_embeds,
                graph_position_features=encoder_graph_position_features,
                orthonormal_features=encoder_orthonormal_features,
                graph_embedding_ids=encoder_graph_embedding_ids,
            )
            fingerprint_tokens = encoder_outputs.fingerprint_tokens
            if self.config.share_embeddings:
                decoder_identifier_embeds = encoder_outputs.identifier_embeds
            encoder_graph_position_embeds = encoder_outputs.graph_position_embeds
            encoder_graph_position_features = encoder_outputs.graph_position_features
            encoder_orthonormal_features = encoder_outputs.orthonormal_features
            encoder_graph_embedding_ids = encoder_outputs.graph_embedding_ids
            values_properties = encoder_outputs.values_properties  # 🔍
            indices_properties = encoder_outputs.indices_properties  # 🔍

        # decoder
        decoder_outputs: GraphsGPTDecoderOutputWithPast = self.decoder(
            decoder_input_ids,
            graph_position_ids_1,
            graph_position_ids_2,
            identifier_ids,
            fingerprint_tokens=fingerprint_tokens,  # 🔍 encoder output fused with property & scaffold info
            attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            identifier_embeds=decoder_identifier_embeds,  # maybe encoder output
            graph_position_embeds=decoder_graph_position_embeds,
            graph_position_features=decoder_graph_position_features,
            orthonormal_features=decoder_orthonormal_features,
            graph_embedding_ids=decoder_graph_embedding_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            past_query_key_values=past_query_key_values,
            use_cache=use_cache,
        )
        hidden_states = decoder_outputs.hidden_states
        inputs_embeds = decoder_outputs.inputs_embeds
        identifier_embeds = decoder_outputs.identifier_embeds
        decoder_graph_position_embeds = decoder_outputs.graph_position_embeds
        decoder_graph_position_features = decoder_outputs.graph_position_features
        decoder_orthonormal_features = decoder_outputs.orthonormal_features
        decoder_graph_embedding_ids = decoder_outputs.graph_embedding_ids
        next_cache = decoder_outputs.past_query_key_values
        all_hidden_states = decoder_outputs.all_hidden_states
        all_self_attns = decoder_outputs.all_attentions

        return GraphsGPTModelConditionedOutputWithPast(
            hidden_states=hidden_states,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            encoder_graph_position_embeds=encoder_graph_position_embeds,
            encoder_graph_position_features=encoder_graph_position_features,
            encoder_orthonormal_features=encoder_orthonormal_features,
            encoder_graph_embedding_ids=encoder_graph_embedding_ids,
            decoder_graph_position_embeds=decoder_graph_position_embeds,
            decoder_graph_position_features=decoder_graph_position_features,
            decoder_orthonormal_features=decoder_orthonormal_features,
            decoder_graph_embedding_ids=decoder_graph_embedding_ids,
            fingerprint_tokens=fingerprint_tokens,
            values_properties=values_properties,  # 🔍
            indices_properties=indices_properties,  # 🔍
            num_fingerprint_tokens=num_fingerprint_tokens,
            all_hidden_states=all_hidden_states,
            all_attentions=all_self_attns,
            past_query_key_values=next_cache,
        )


class GraphsGPTForConditionalGeneration(GraphsGPTConditionedPreTrainedModel, GraphsGPTForCausalLM):
    """Model for finetuning by adding conditional properties to the decoder inputs."""

    def __init__(self, config: GraphsGPTConditionedConfig):
        super(GraphsGPTForCausalLM, self).__init__(config)
        self.model = GraphsGPTModelConditioned(config)  # 🔍

        # next bond token prediction head
        self.lm_head = nn.Linear(config.hidden_size, config.bond_vocab_size + 1, bias=False)  # EOS token at position 0

        # first atom token prediction head
        self.n_head = nn.Linear(config.hidden_size, config.atom_vocab_size, bias=False)

        # connection prediction heads
        self.c_head_begin = nn.Linear(config.hidden_size, config.position_feature_size, bias=True)
        self.c_head_end = nn.Linear(config.hidden_size, config.position_feature_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            graph_position_ids_1: torch.LongTensor = None,
            graph_position_ids_2: torch.LongTensor = None,
            identifier_ids: torch.BoolTensor = None,
            attention_mask: Optional[torch.Tensor] = None,  # padding mask

            scaffold_input_ids: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_1: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_2: torch.LongTensor = None,  # 🔍
            scaffold_identifier_ids: torch.BoolTensor = None,  # 🔍
            scaffold_attention_mask: Optional[torch.Tensor] = None,  # 🔍 padding mask

            values_qed: torch.FloatTensor = None,  # 🔍
            values_sa: torch.FloatTensor = None,  # 🔍
            values_logp: torch.FloatTensor = None,  # 🔍

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
            scaffold_input_ids=scaffold_input_ids,  # 🔍
            scaffold_graph_position_ids_1=scaffold_graph_position_ids_1,  # 🔍
            scaffold_graph_position_ids_2=scaffold_graph_position_ids_2,  # 🔍
            scaffold_identifier_ids=scaffold_identifier_ids,  # 🔍
            scaffold_attention_mask=scaffold_attention_mask,  # 🔍
            values_qed=values_qed,  # 🔍
            values_sa=values_sa,  # 🔍
            values_logp=values_logp,  # 🔍
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
            scaffold_input_ids: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_1: torch.LongTensor = None,  # 🔍
            scaffold_graph_position_ids_2: torch.LongTensor = None,  # 🔍
            scaffold_identifier_ids: torch.BoolTensor = None,  # 🔍
            scaffold_attention_mask: Optional[torch.Tensor] = None,  # 🔍 padding mask

            values_qed: torch.FloatTensor = None,  # 🔍
            values_sa: torch.FloatTensor = None,  # 🔍
            values_lipinski: torch.FloatTensor = None,  # 🔍
            values_logp: torch.FloatTensor = None,  # 🔍

            num_fingerprint_tokens: Optional[int] = None,

            property_embeds: Optional[torch.FloatTensor] = None,  # 🔍
            inputs_embeds: Optional[torch.FloatTensor] = None,
            identifier_embeds: Optional[torch.FloatTensor] = None,
            graph_position_embeds: Optional[torch.FloatTensor] = None,
            graph_position_features: Optional[torch.FloatTensor] = None,
            orthonormal_features: Optional[torch.FloatTensor] = None,
            graph_embedding_ids: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> torch.FloatTensor:
        encoder_outputs: GraphsGPTEncoderConditionedOutput = self.model.encoder(
            scaffold_input_ids,
            scaffold_graph_position_ids_1,
            scaffold_graph_position_ids_2,
            scaffold_identifier_ids,
            values_qed,
            values_sa,
            values_logp,
            num_fingerprint_tokens=num_fingerprint_tokens,
            attention_mask=scaffold_attention_mask,
            property_embeds=property_embeds,
            inputs_embeds=inputs_embeds,
            identifier_embeds=identifier_embeds,
            graph_position_embeds=graph_position_embeds,
            graph_position_features=graph_position_features,
            orthonormal_features=orthonormal_features,
            graph_embedding_ids=graph_embedding_ids,
        )
        fingerprint_tokens = encoder_outputs.fingerprint_tokens

        return fingerprint_tokens
