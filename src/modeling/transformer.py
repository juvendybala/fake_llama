from typing import List, Optional, Tuple, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

from .config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)


@config_dataclass
class TransformerConfig(BaseConfig):
    """Transformer Configurations Dataclass"""
    
    # common transformer configurations
    num_layers: int = make_required_field()
    hidden_size: int = make_required_field()
    ffh_size: int = make_required_field()
    max_seq_len: int = make_required_field()
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = make_fixed_field(0)
    world_size: int = make_fixed_field(1)
    process_group: Optional[ProcessGroup] = make_fixed_field(None)
    
    # vocab embedding configurations
    vocab_size: int = make_required_field()
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = make_required_field()
    num_q_head: int = make_required_field()
    num_kv_head: int = make_required_field()
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        super().__init__()
        self.layout = qkv_layout
        self.num_layers = num_layers
        self.is_exist = [False for _ in range(num_layers)]
        self.key_cache =  [None for _ in range(num_layers)]
        self.value_cache = [None for _ in range(num_layers)]
        if self.layout == AttnQKVLayout.THD:
            self.cu_seqlen_cache = [None for _ in range(num_layers)]

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """
        if layer_idx >= self.num_layers or self.is_exist[layer_idx] == False:
            return False
        else:
            return True

    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        if self.has(layer_idx) == False:
            raise KeyError
        else:
            if(self.layout == AttnQKVLayout.THD):
                return self.key_cache[layer_idx], self.value_cache[layer_idx], self.cu_seqlen_cache[layer_idx]
            else:
                return self.key_cache[layer_idx], self.value_cache[layer_idx], None

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        if self.layout != AttnQKVLayout.THD:
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
        else:
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v
            self.cu_seqlen_cache[layer_idx] = cu_seqlens
        self.is_exist[layer_idx] = True

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        if self.has(layer_idx) == False:
            self.set(layer_idx=layer_idx, k=k, v=v, cu_seqlens=cu_seqlens)
        else:
            if(self.layout == AttnQKVLayout.BSHD):
                # print(layer_idx)
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=-3)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=-3)
            elif(self.layout == AttnQKVLayout.SBHD):
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], k], dim=-4)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], v], dim=-4)
            else:
                # print(cu_seqlens.shape)
                tmp_k = []
                tmp_v = []
                for i in range(cu_seqlens.shape[0] - 1):
                    tmp_k.append(torch.cat([self.key_cache[layer_idx][self.cu_seqlen_cache[layer_idx][i] : self.cu_seqlen_cache[layer_idx][i+1]], k[cu_seqlens[i] : cu_seqlens[i+1]]], dim=0))
                    tmp_v.append(torch.cat([self.value_cache[layer_idx][self.cu_seqlen_cache[layer_idx][i] : self.cu_seqlen_cache[layer_idx][i+1]], v[cu_seqlens[i] : cu_seqlens[i+1]]], dim=0))
                self.key_cache[layer_idx] = torch.cat(tmp_k, dim=0)
                self.value_cache[layer_idx] = torch.cat(tmp_v, dim=0)
                self.cu_seqlen_cache[layer_idx] = self.cu_seqlen_cache[layer_idx] + cu_seqlens
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        self.is_exist = [False for _ in range(self.num_layers)]
        self.key_cache =  [None for _ in range(self.num_layers)]
        self.value_cache = [None for _ in range(self.num_layers)]
        if self.layout == AttnQKVLayout.THD:
            self.cu_seqlen_cache = [None for _ in range(self.num_layers)]
    

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        self.config = config
        self.layer_idx = layer_idx
        self.init = False
        self.attn_norm = GroupRMSNorm(hidden_size=config.hidden_size,
                                      group_size=config.group_size,
                                      eps=config.eps,
                                      init_range=config.norm_init_range,
                                      init_seed=config.init_base_seed + layer_idx + 1,
                                      dtype=config.param_dtype,
                                      device=config.param_device)
        
        self.mlp_norm = GroupRMSNorm(hidden_size=config.hidden_size,
                                     group_size=config.group_size,
                                     eps=config.eps,
                                     init_range=config.norm_init_range,
                                     init_seed=config.init_base_seed+layer_idx+3,
                                     dtype=config.param_dtype,
                                     device=config.param_device)
        
        self.rotary_emb = NTKAwareRoPE(dim=config.head_dim,
                                       max_seq_len=config.max_seq_len,
                                       base=config.rope_base,
                                       ratio=config.rope_ratio,
                                       dynamic=config.rope_dynamic,
                                       dtype=config.param_dtype,
                                       device=config.param_device)
        
        if config.num_experts is None:
            self.mlp = DenseMLPWithLoRA(hidden_size=config.hidden_size,
                                        ffh_size=config.ffh_size,
                                        activation_type=config.activation_type,
                                        init_base_seed=config.init_base_seed + layer_idx + 4,
                                        lora_rank=config.lora_rank,
                                        lora_alpha=config.lora_alpha,
                                        lora_dropout_rate=config.lora_dropout_rate,
                                        lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                                        lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                                        dtype=config.param_dtype,
                                        device=config.param_device)
        else:
            self.mlp = SparseMLPWithLoRA(hidden_size=config.hidden_size,
                                         ffh_size=config.ffh_size,
                                         activation_type=config.activation_type,
                                         num_experts=config.num_experts,
                                         moe_topk=config.moe_topk,
                                         rank=config.rank,
                                         world_size=config.world_size,
                                         process_group=config.process_group,
                                         init_mean=config.gate_init_mean,
                                         init_std=config.gate_init_std,
                                         init_base_seed=config.init_base_seed + layer_idx + 4,
                                         lora_rank=config.lora_rank,
                                         lora_alpha=config.lora_alpha,
                                         lora_dropout_rate=config.lora_dropout_rate,
                                         lora_dropout_seed=config.lora_dropout_seed + layer_idx,
                                         lora_init_base_seed=config.lora_init_base_seed + layer_idx,
                                         dtype=config.param_dtype,
                                         device=config.param_device)
            
        if config.online_attn_block_size is None:
            self.attn = OfflineSlidingWindowAttn(head_dim=config.head_dim,
                                                 num_q_head=config.num_q_head,
                                                 num_kv_head=config.num_kv_head,
                                                 qkv_pack_format=config.qkv_pack_format,
                                                 qkv_layout=config.qkv_layout,
                                                 window_size=config.window_size,
                                                 causal=config.causal,
                                                 softmax_dropout_rate=config.softmax_dropout_rate,
                                                 softmax_dropout_seed=config.softmax_dropout_seed + layer_idx,
                                                 softmax_scale=config.softmax_scale,
                                                 softmax_cap=config.softmax_cap,
                                                 softmax_temp=config.softmax_temp,
                                                 softmax_clip_range=config.softmax_clip_range,
                                                 apply_qk_norm=config.apply_qk_norm,
                                                 group_size=config.qk_norm_group_size,
                                                 eps=config.eps,
                                                 init_range=config.norm_init_range,
                                                 init_seed=config.init_base_seed + layer_idx + 2,
                                                 dtype=config.param_dtype,
                                                 device=config.param_device)
        else:
            self.attn = OnlineSlidingWindowAttn(seqlen_q=config.max_seq_len,
                                                seqlen_kv=config.max_seq_len,
                                                block_size_q=config.online_attn_block_size,
                                                block_size_kv=config.online_attn_block_size,
                                                head_dim=config.head_dim,
                                                num_q_head=config.num_q_head,
                                                num_kv_head=config.num_kv_head,
                                                window_size=config.window_size,
                                                causal=config.causal,
                                                softmax_scale=config.softmax_scale,
                                                softmax_cap=config.softmax_cap,
                                                softmax_temp=config.softmax_temp,
                                                apply_qk_norm=config.apply_qk_norm,
                                                group_size=config.qk_norm_group_size,
                                                eps=config.eps,
                                                init_range=config.norm_init_range,
                                                init_seed=config.init_base_seed + layer_idx + 2,
                                                dtype=config.param_dtype,
                                                device=config.param_device)
        self.reset_parameters()
        self.init = True
    
    def qkv_generate(self, input: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_dim = self.config.head_dim
        nq = self.config.num_q_head
        nkv = self.config.num_kv_head
        q_size = nq * head_dim
        kv_size = nkv * head_dim
        qkv = torch.matmul(input, self.qkv_proj)
        q = qkv.narrow(dim=-1, start=0, length=q_size)
        k = qkv.narrow(dim=-1, start=q_size, length=kv_size)
        v = qkv.narrow(dim=-1, start=q_size + kv_size, length=kv_size)
        b, s, hd = input.shape
        # assert(nq != 0)
        # assert(nkv != 0)
        q = q.view(b, s, nq, head_dim)
        k = k.view(b, s, nkv, head_dim)
        v = v.view(b, s, nkv, head_dim)
        q = self.rotary_emb(q, offset)
        k = self.rotary_emb(k, offset)
        return q, k, v
    
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        input_device = input.device
        input_dtype = input.dtype
        input = input.to(device=self.config.param_device, dtype=self.config.param_dtype)
        residual = input
        hidden = self.attn_norm(input)
        b, s, hd = hidden.shape
        if kv_cache is not None and kv_cache.has(self.layer_idx):
            cached_k, cached_v, cached_cu_kv_seqlens = kv_cache.get(self.layer_idx)
        if self.config.qkv_layout == AttnQKVLayout.BSHD:
            if kv_cache is not None and kv_cache.has(self.layer_idx):
                offset = cached_k.shape[1]
            else:
                offset = 0
            q, k, v = self.qkv_generate(hidden, offset)
            if kv_cache is not None:
                kv_cache.append(self.layer_idx, k, v)
                k, v, cu_kv_seqlens = kv_cache.get(self.layer_idx)
        elif self.config.qkv_layout == AttnQKVLayout.SBHD:
            if kv_cache is not None and kv_cache.has(self.layer_idx):
                offset = cached_k.shape[0]
            else:
                offset = 0
            q, k, v = self.qkv_generate(hidden, offset)
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            if kv_cache is not None:
                kv_cache.append(self.layer_idx, k, v)
                k, v, cu_kv_seqlens = kv_cache.get(self.layer_idx)
        else:
            for i in range(cu_seqlens.shape[0] - 1):
                if kv_cache is not None and kv_cache.has(self.layer_idx):
                    offset = cached_cu_kv_seqlens[i+1] - cached_cu_kv_seqlens[i]
                else:
                    offset = 0
                q_tmp, k_tmp, v_tmp = self.qkv_generate(hidden[:, cu_seqlens[i]:cu_seqlens[i+1], :], offset)
                if i == 0:
                    q, k ,v = q_tmp, k_tmp, v_tmp
                else:
                    q = torch.cat([q, q_tmp], dim=1)
                    k = torch.cat([k, k_tmp], dim=1)
                    v = torch.cat([v, v_tmp], dim=1)
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)
            if kv_cache is not None:
                kv_cache.append(self.layer_idx, k, v, cu_seqlens)
                k, v, cu_kv_seqlens = kv_cache.get(self.layer_idx)
       
        if self.config.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            k = torch.cat([k, v], dim=-2)
        elif self.config.qkv_pack_format == AttnQKVPackFormat.QKV:
            q = torch.cat([q, k, v], dim=-2)
        if self.config.online_attn_block_size is not None:
            kv_s = k.shape[1]
            block_size = self.config.online_attn_block_size
            q_num = (s + block_size - 1) // block_size
            kv_num = (kv_s + block_size - 1) // block_size
            q_len = block_size * q_num
            kv_len = block_size * kv_num
            q = F.pad(input=q, pad=(0, 0, 0, q_len - s), mode="constant", value=0)
            k = F.pad(input=k, pad=(0, 0, 0, kv_len - kv_s), mode="constant", value=0)
            v = F.pad(input=v, pad=(0, 0, 0, kv_len - kv_s), mode="constant", value=0)
            global_o = torch.zeros((b, q_len, self.config.num_q_head, self.config.head_dim),
                                   dtype=q.dtype,
                                   device=q.device)
            global_lse = torch.full((b, self.config.num_q_head, q_len),
                                    fill_value=-float('inf'),
                                    dtype=torch.float32,
                                    device=q.device)
            for i in range(q_num):
                start_q = i * block_size
                end_q = (i + 1) * block_size
                q_block = q[:, start_q:end_q, :, :]
                for j in range(kv_num):
                    start_kv = j * block_size
                    end_kv = (j + 1) * block_size
                    k_block = k[:, start_kv:end_kv, :, :]
                    v_block = v[:, start_kv:end_kv, :, :]
                    self.attn(q_block, k_block, v_block, global_o, global_lse, i, j)
            attn_output = global_o[:, :s, :]
        else:
            if self.config.qkv_layout != AttnQKVLayout.THD:
                attn_output = self.attn(q, k, v)
            else:
                if kv_cache is not None:
                    attn_output = self.attn(q, k, v, cu_seqlens, cu_kv_seqlens)
                else:
                    attn_output = self.attn(q, k, v, cu_seqlens, cu_seqlens)
        if self.config.qkv_layout == AttnQKVLayout.SBHD:
            attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(b, s, -1)
        attn_output = torch.matmul(attn_output, self.o_proj)
        hidden = residual + attn_output
        residual = hidden
        hidden = self.mlp_norm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden.to(dtype=input_dtype, device=input_device)
    
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Layer module"""
        # raise NotImplementedError("TODO: Assignment4 - Task2")
        qkv_size = (self.config.num_q_head + 2 * self.config.num_kv_head) * self.config.head_dim
        self.qkv_proj = nn.Parameter(torch.empty((self.config.hidden_size, qkv_size), device=self.config.param_device, dtype=self.config.param_dtype))
        torch.manual_seed(self.config.proj_init_seed + self.layer_idx + 1)
        nn.init.normal_(self.qkv_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        o_size = self.config.num_q_head * self.config.head_dim
        self.o_proj = nn.Parameter(torch.empty((o_size, self.config.hidden_size), dtype=self.config.param_dtype, device=self.config.param_device))
        torch.manual_seed(self.config.proj_init_seed + self.layer_idx + 2)
        nn.init.normal_(self.o_proj, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
        if self.init:
            self.attn_norm.reset_parameters()
            self.mlp_norm.reset_parameters()
            self.mlp.reset_parameters()


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        self.init = False
        self.vocab_emb = ParallelVocabEmbedding(
            vocab_size=config.vocab_size,
            emb_size=config.hidden_size,
            rank=config.rank,
            world_size=config.world_size,
            process_group=config.process_group,
            init_mean=config.vocab_init_mean,
            init_std=config.vocab_init_std,
            init_base_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        self.final_norm = GroupRMSNorm(
            hidden_size=config.hidden_size,
            group_size=config.group_size,
            eps=config.eps,
            init_range=config.norm_init_range,  
            init_seed=config.init_base_seed,
            dtype=config.param_dtype,
            device=config.param_device
        )
        self.kv_cache = TransformerDecoderKVCache(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers
        )
        self.config = config
        self.reset_parameters()
        self.init = True
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        input_device = input_ids.device
        input_dtype = input_ids.dtype
        input_ids = input_ids.to(device=self.config.param_device)
        hidden_states = self.vocab_emb(input_ids)
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                kv_cache=self.kv_cache if not self.training else None
            )
        hidden_states = self.final_norm(hidden_states)
        if self.config.lm_head_tied:
            logits = torch.matmul(hidden_states, self.vocab_emb.tr.T)
        else:
            logits = self.lm_head(hidden_states)
            # logits = torch.matmul(hidden_states, self.lm_head.T)
        # logits = self.lm_head(hidden_states)
        return logits.to(device=input_device)
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        return self.kv_cache
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        self.kv_cache = kv_cache
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        self.kv_cache.reset()
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        if not self.config.lm_head_tied:            
            self.lm_head = nn.Linear(
                in_features=self.config.hidden_size,
                out_features=self.config.vocab_size,
                bias=False,
                device=self.config.param_device,
                dtype=self.config.param_dtype
            )
            torch.manual_seed(self.config.proj_init_seed)
            nn.init.normal_(tensor=self.lm_head.weight, mean=self.config.proj_init_mean, std=self.config.proj_init_std)
            # self.lm_head = nn.Parameter(
            #     torch.empty(
            #         self.config.vocab_size, 
            #         self.config.hidden_size,
            #         dtype=self.config.param_dtype,
            #         device=self.config.param_device
            #     )
            # )
            # nn.init.normal_(
            #     self.lm_head,
            #     mean=self.config.proj_init_mean,
            #     std=self.config.proj_init_std
            # )
            # print(self.lm_head.weight.data)
        # else:
        #     self.lm_head = nn.Linear(
        #         in_features=self.config.hidden_size,
        #         out_features=self.config.vocab_size,
        #         bias=False,
        #         device=self.config.param_device,
        #         dtype=self.config.param_dtype
        #     )
        #     self.lm_head.weight = self.vocab_emb.tr
        if self.init == True:
            self.vocab_emb.reset_parameters()
            self.final_norm.reset_parameters()
            # for decoder_layer in self.decoder_layers:
            #     decoder_layer.reset_parameters()
     
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
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        total_params = 0
        for p in self.parameters():
            if learnable_only == False or p.requires_grad == True:
                total_params += p.numel()
        if unit == "1":
            return total_params / 1.0
        elif unit == "K":
            return total_params / 1e3
        elif unit == "M":
            return total_params / 1e6
        else:
            return total_params / 1e9

    
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
        # raise NotImplementedError("TODO: Assignment4 - Task3")
        total_bytes = 0
        for p in self.parameters():
            if p.dtype == torch.float32:
                total_bytes += p.numel() * 4
            elif p.dtype == torch.float16 or p.dtype == torch.bfloat16:
                total_bytes += p.numel() * 2
            elif p.dtype == torch.float64:
                total_bytes += p.numel() * 8
            else:
                assert(1 == 0)
        if unit == "B":
            return total_bytes / 1.0
        elif unit == "KB":
            return total_bytes / 1024
        elif unit == "MB":
            return total_bytes / 1024 ** 2
        else:
            return total_bytes / 1024 ** 3