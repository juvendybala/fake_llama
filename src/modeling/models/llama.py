from typing import Optional, Tuple, Union, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

# from assignment2 implementations
from ..mlp import MLPActivationType

# from assignment3 implementations
from ..attention import AttnQKVLayout, AttnQKVPackFormat

# from assignment4 implementations
from ..transformer import (
    TransformerConfig,
    TransformerDecoderKVCache,
    TransformerDecoderBlock,
)

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ...utils import (
    convert_to_list,
    load_json,
    load_safetensors,
)

from .base import BaseTokenizer, BaseModel


@config_dataclass
class LlamaConfig(TransformerConfig):
    """Llama Configurations Dataclass
    NOTE: we will fix some configurations unavailable in the original Llama config or troublesome to handle with the pretrained params, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py#L26)
    """
    
    # fixed rope configurations
    rope_ratio: int = make_fixed_field(1)
    rope_dynamic: bool = make_fixed_field(False)
    
    # fixed normalization configurations
    group_size: Optional[int] = make_fixed_field(None)
    
    # fixed attention configurations
    qkv_pack_format: AttnQKVPackFormat = make_fixed_field(AttnQKVPackFormat.Q_K_V)
    qkv_layout: AttnQKVLayout = make_fixed_field(AttnQKVLayout.BSHD)
    window_size: Optional[int] = make_fixed_field(None)
    causal: bool = make_fixed_field(True)
    softmax_scale: Optional[float] = make_fixed_field(None)
    softmax_cap: Optional[float] = make_fixed_field(None)
    softmax_temp: float = make_fixed_field(1.0)
    softmax_clip_range: Tuple[float, float] = make_fixed_field((0., 1.))
    apply_qk_norm: bool = make_fixed_field(False)
    qk_norm_group_size: Optional[int] = make_fixed_field(None)
    
    # fixed sparse mlp configurations
    num_experts: Optional[int] = make_fixed_field(None)
    moe_topk: int = make_fixed_field(1)
    

class LlamaTokenizer(BaseTokenizer):
    """Llama Tokenizer module
    This is a tokenizer module that holds an instance of `Tokenizer` from `tokenizers` and loads its pretrained vocabulary and configuration, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/tokenization_llama.py#L54)
    with which this module implements the common APIs of `BaseTokenizer`
    """
    
    def __init__(
        self,
        vocab_file: str,
        config_file: str,
    ):
        """Initialize Llama Tokenizer module
        
        Args:
            vocab_file (str): path to the file with the pretrained vocabulary in .json format
            config_file (str): path to the file with the configuration in .json format
        """
        super().__init__()
        
        self.vocab_file = vocab_file
        self.config_file = config_file
        
        self._tokenizer = Tokenizer.from_file(vocab_file)
        
        self.config = load_json(config_file)
        
        self._bos_token = self.config["bos_token"]
        self._eos_token = self.config["eos_token"]
        self._pad_token = self.config.get("pad_token", self._eos_token)
        self._bos_token_id, self._eos_token_id, self._pad_token_id = [
            self._tokenizer.token_to_id(token)
            for token in [self._bos_token, self._eos_token, self._pad_token]
        ]
    
    def encode(
        self,
        prompt: Union[str, List[str]],
    ) -> List[torch.LongTensor]:
        """Encode a single or a list of prompt(s) into a list of tensors of encoded ids
        
        Args:
            prompt (Union[str, List[str]]): a single or a list of prompt(s) to be encoded
        
        Returns:
            List[torch.LongTensor]: a list of 1-dim tensors of encoded ids, each with shape: [seq_len,], \
                NOTE: if the input is a single prompt, the output will be a list with length 1
        """
        
        # convert to prompt list
        prompt = convert_to_list(prompt)
        
        # encode each prompt
        encoded_ids = [
            torch.tensor(
                encoded.ids,
                dtype=torch.long,
            )
            for encoded in self._tokenizer.encode_batch(prompt)
        ]
            
        return encoded_ids
    
    def decode(
        self,
        encoded_ids: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> List[str]:
        """Decode a single or a list of tensor(s) of encoded ids into a list of prompts
        
        Args:
            encoded_ids (Union[torch.LongTensor, List[torch.LongTensor]]): a single or a list of tensor(s) of encoded ids to be decoded, \
                NOTE: each tensor is a 1-dim tensor with shape: [seq_len,]
        
        Returns:
            List[str]: a list of prompt strings, each with length: `seq_len` (stripped the special tokens), \
                NOTE: if the input is a single tensor of encoded ids, the output will be a list with length 1
        """
        
        # convert to encoded ids list
        encoded_ids = convert_to_list(encoded_ids)
        encoded_ids = [encoded.tolist() for encoded in encoded_ids]
        
        # decode each encoded ids
        return self._tokenizer.decode_batch(encoded_ids)
    
    @property
    def vocab_size(self) -> int:
        """The vocab size of the tokenizer"""
        return self._tokenizer.get_vocab_size()
    
    @property
    def pad_token(self) -> str:
        """The pad token of the tokenizer"""
        return self._pad_token
    
    @property
    def pad_id(self) -> int:
        """The pad token id of the tokenizer"""
        return self._pad_token_id
    
    @property
    def bos_token(self) -> str:
        """The bos token of the tokenizer"""
        return self._bos_token
    
    @property
    def bos_id(self) -> int:
        """The bos token id of the tokenizer"""
        return self._bos_token_id
    
    @property
    def eos_token(self) -> str:
        """The eos token of the tokenizer"""
        return self._eos_token
    
    @property
    def eos_id(self) -> int:
        """The eos token id of the tokenizer"""
        return self._eos_token_id


class LlamaModel(BaseModel):
    """Llama Model module
    This is a variant modeling of the `LlamaForCausalLM` compatible with the pretrained parameters, \
        (See: https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L1105)
    which supports the common APIs of `BaseModel` and the extra loading methods \
        to load configurations and pretrained weights from the original HF repo
    """
    
    def __init__(
        self,
        config: LlamaConfig,
    ):
        """Initialize Llama Model module
        
        Args:
            config (LlamaConfig): Llama configuration
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        self.config = config
        self.transformer_decoder = TransformerDecoderBlock(config=config)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """The forward pass of the Llama Model module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, either as a training sample or as a query prompt, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
            labels(Optional[torch.LongTensor]): the vocab ids for the training labels, with the same shape as `input_ids` (and the `ignore_index` is set to `-100` as default by pytorch)
            NOTE: due to the nll loss kernel restriction, the dtype of labels should be exactly int64 (torch.long) if provided
            temperature (float, optional): temperature for softmax during inference, default to 1.0, in the range of (0., +inf)
        Returns:
            torch.Tensor: output tensor, whose value depends on the model's state:
                1. for training, the value is just the cross-entroy scalar loss by `mean` reduction between the vocab logits and the given labels
                2. for inference, the value is the next-token probability distribution for every sequence in the batch, with shape: [inferred_batch_size, vocab_size]
            NOTE: the `inferred_batch_size` is inferred from `cu_seqlens` if provided (i.e. `inner_batch_size`), otherwise from `input_ids` (i.e. `batch_size`)
        """
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        if self.training:
            logits = self.transformer_decoder(input_ids, cu_seqlens)
            b, s, v = logits.shape
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(b * s, -1), labels.view(b * s))
            return loss
        else:
            with torch.no_grad():
                logits = self.transformer_decoder(input_ids, cu_seqlens)
                scaled_logits = logits / temperature
                softmax_output = F.softmax(input=scaled_logits, dim=-1, dtype=torch.float32)
                if cu_seqlens is None:
                    return softmax_output[:, -1, :]
                else:
                    result = []
                    for i in range(cu_seqlens.shape[0] - 1):
                        result.append(softmax_output[:, cu_seqlens[i+1] - 1, :])
                    return torch.cat(result, dim=0)
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        return self.transformer_decoder.get_kv_cache()
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        self.transformer_decoder.set_kv_cache(kv_cache)
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        self.transformer_decoder.reset_kv_cache()
    
    def reset_parameters(self):
        """Initialize learnable parameters for Llama Model module"""
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        self.transformer_decoder.reset_parameters()
    
    def load_parameters(self, params_files: Union[str, List[str]]) -> None:
        """Load pretrained params from the original LlamaForCausalLM
        
        Args:
            params_files(Union[str, List[str]]): path to the pretrained params file(s) of the original LlamaForCausalLM in .safetensors format
                NOTE: if there're multiple params files, it is ensured that there is NO param name conflict
        """
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        params_dict = load_safetensors(params_files)
        load_dict = {}
        # print(params_dict)
        # assert 1 == 0
        # for name, parameter in self.named_parameters():
        #     print(f"{name} : {parameter.shape}")
        # for name, weight in params_dict.items():
        #     print(f"{name} : {weight.shape}")
        for name, weight in params_dict.items():
            if "embed_tokens" in name:
                load_dict["transformer_decoder.vocab_emb.tr"] = weight
            elif name == "model.norm.weight":
                load_dict["transformer_decoder.final_norm.weight"] = weight
            elif "input_layernorm" in name:
                name = name.replace("model", "transformer_decoder")
                name = name.replace("layers", "decoder_layers")
                name = name.replace("input_layernorm", "attn_norm")
                load_dict[name] = weight
            elif "post_attention_layernorm" in name:
                name = name.replace("model", "transformer_decoder")
                name = name.replace("layers", "decoder_layers")
                name = name.replace("post_attention_layernorm", "mlp_norm")
                load_dict[name] = weight
            elif "o_proj" in name:
                name = name.replace("model", "transformer_decoder")
                name = name.replace("layers", "decoder_layers")
                name = name.replace("self_attn.o_proj.weight", "o_proj")
                load_dict[name] = weight.T
            elif "mlp" in name:
                name = name.replace("model", "transformer_decoder")
                name = name.replace("layers", "decoder_layers")
                if "up" in name:
                    name = name.replace("up_proj.weight", "up")
                elif "gate" in name:
                    name = name.replace("gate_proj.weight", "gate")
                elif "down" in name:
                    name = name.replace("down_proj.weight", "down")
                load_dict[name] = weight.T
        for i in range(self.config.num_layers):
            for name, weight in params_dict.items():
                if str(i) in name and "q_proj" in name:
                    q_weight = weight
                if str(i) in name and "k_proj" in name:
                    k_weight = weight
                if str(i) in name and "v_proj" in name:
                    v_weight = weight
            load_dict["transformer_decoder.decoder_layers." + str(i) + ".qkv_proj"] = torch.cat([q_weight, k_weight, v_weight], dim=0).T
        # for name, weight in load_dict.items():
        #     print(f"{name} : {weight.shape}")
        self.load_state_dict(load_dict)
    
    @staticmethod
    def load_config(config_file: str, **extra_configs) -> LlamaConfig:
        """Load config from the original original Llama config
        
        Args:
            config_file(str): path to the config file of the original original Llama config in .json format
            extra_configs(dict, optional): extra (key, value) config pair(s), to overwrite `config.key = value`, \
                helpful to set some configurations that are neither fixed nor provided in the original config such as `device`, `seed`, etc.
                NOTE: if any required configuration is not found in the original config, you are supposed to pass it in `extra_configs`, \
                    otherwise, a `ValueError` will be raised.
        Returns:
            LlamaConfig: a LlamaConfig object initialized from the config file
        """
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        config_dict = load_json(config_file)
        config_dict.update(extra_configs)
        if "hidden_size" not in config_dict or "num_hidden_layers" not in config_dict or "intermediate_size" not in config_dict or ("rope_scaling" not in config_dict and "original_max_position_embeddings" not in config_dict["rope_scaling"]) or\
        "vocab_size" not in config_dict or "head_dim" not in config_dict or "num_attention_heads" not in config_dict or "num_key_value_heads" not in config_dict:
            raise ValueError
        if config_dict["hidden_act"] == "silu":
            activation_type = MLPActivationType.SILU
        if config_dict["torch_dtype"] == "bfloat16":
            torch_dtype = torch.bfloat16
        llamaconfig = LlamaConfig(
            num_layers = config_dict["num_hidden_layers"],
            hidden_size = config_dict["hidden_size"],
            ffh_size = config_dict["intermediate_size"],
            max_seq_len = config_dict["rope_scaling"]["original_max_position_embeddings"],
            vocab_size = config_dict["vocab_size"],
            head_dim = config_dict["head_dim"],
            num_q_head = config_dict["num_attention_heads"],
            num_kv_head = config_dict["num_key_value_heads"],
            softmax_dropout_rate = config_dict["attention_dropout"],
            activation_type = activation_type,
            eps =config_dict["rms_norm_eps"],
            lm_head_tied = config_dict["tie_word_embeddings"],
            param_dtype = torch_dtype,
            rope_base = int(config_dict["rope_theta"])
        )
        return llamaconfig
    
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        return self.transformer_decoder.num_parameters(learnable_only, unit)
    
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        # raise NotImplementedError("TODO: Assignment5 - Task1")
        return self.transformer_decoder.num_memory_footprint(unit)
    