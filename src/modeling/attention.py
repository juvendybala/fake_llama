from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        self.hd = head_dim
        self.nq = num_q_head
        self.nkv = num_kv_head
        self.qkv_pack_format = qkv_pack_format
        self.qkv_layout = qkv_layout
        self.window_size = window_size
        self.casual = causal
        self.softmax_dropout_rate = softmax_dropout_rate
        self.softmax_dropout_seed = softmax_dropout_seed
        self.softmax_scale = softmax_scale
        if softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hd)
        self.softmax_cap = softmax_cap
        self.softmax_temp = softmax_temp
        self.softmax_clip_range = softmax_clip_range
        self.apply_qk_norm = apply_qk_norm
        if apply_qk_norm:
            if group_size is None:
                group_size = self.hd
            # print(group_size)
            self.normq = GroupRMSNorm(self.hd * self.nq, group_size, eps, init_range, init_seed, dtype, device)
            self.normk = GroupRMSNorm(self.hd * self.nkv, group_size, eps, init_range, init_seed, dtype, device)
    
    def process(self, chunk_q: torch.Tensor, chunk_k: torch.Tensor, chunk_v: torch.Tensor) -> torch.Tensor:
        bs_q, sq, nq, hd_q = chunk_q.shape
        bs_kv, skv, nkv, hd_kv = chunk_k.shape
        assert(nq == self.nq and nkv == self.nkv)
        assert(bs_q == bs_kv)
        assert(self.hd == hd_q and self.hd == hd_kv)
        if self.apply_qk_norm:
            # assert(self.nq != 0)
            # assert(self.nkv != 0)
            chunk_q = self.normq(chunk_q.reshape(bs_q, sq, self.nq * self.hd)).reshape(bs_q, sq, self.nq, self.hd)
            chunk_k = self.normk(chunk_k.reshape(bs_kv, skv, self.nkv * self.hd)).reshape(bs_kv, skv, self.nkv, self.hd)
        if self.nq != self.nkv:
            chunk_k = chunk_k.repeat_interleave(self.nq // self.nkv, dim=2)
            chunk_v = chunk_v.repeat_interleave(self.nq // self.nkv, dim=2)
        chunk_q = chunk_q.transpose(1, 2)
        chunk_k = chunk_k.transpose(1, 2)
        chunk_v = chunk_v.transpose(1, 2)
        attn_weights = torch.matmul(chunk_q, chunk_k.transpose(-2, -1)) * self.softmax_scale
        if self.softmax_cap is None:
            attn_weights = attn_weights / self.softmax_temp
        else:
            attn_weights = self.softmax_cap * torch.tanh(attn_weights / self.softmax_cap)
        if self.window_size is not None or self.casual:
            max_size = max(sq, skv)
            pos_q = torch.arange(sq, device=chunk_q.device)
            pos_kv = torch.arange(skv, device=chunk_k.device)
            pos_q = pos_q + (max_size - sq)
            pos_kv = pos_kv + (max_size - skv)
            dist = pos_q.unsqueeze(-1) - pos_kv.unsqueeze(0)
            mask = torch.zeros((sq, skv), device=chunk_q.device, dtype=chunk_q.dtype)
            if self.window_size is not None:
                mask = mask.masked_fill(dist.abs() > self.window_size, float('-inf'))
            if self.casual:
                mask = mask.masked_fill(dist < 0, float('-inf'))
            attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)
        # print(mask)
        # print(attn_weights)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.nan_to_num(nan=0.0)
        # print(attn_weights)
        l, r = self.softmax_clip_range
        attn_weights = torch.clamp((r - l) * attn_weights + l, 0, 1)
        torch.manual_seed(self.softmax_dropout_seed)
        attn_weights = F.dropout(attn_weights, p=self.softmax_dropout_rate)
        result = torch.matmul(attn_weights, chunk_v)
        # print(result.shape)
        return result.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        # raise NotImplementedError("Assignment3 - Task1")
        # print(f"input_q : {q.shape}")
        if self.qkv_pack_format == AttnQKVPackFormat.QKV:
            q_tmp = q.narrow(dim=-2, start=0, length=self.nq)
            k_tmp = q.narrow(dim=-2, start=self.nq, length=self.nkv)
            v_tmp = q.narrow(dim=-2, start=self.nq + self.nkv, length=self.nkv)
        elif self.qkv_pack_format == AttnQKVPackFormat.Q_KV:
            q_tmp = q
            k_tmp = k.narrow(dim=-2, start=0, length=self.nkv).to(device=q.device, dtype=q.dtype)
            v_tmp = k.narrow(dim=-2, start=self.nkv, length=self.nkv).to(device=q.device, dtype=q.dtype)
        else:
            q_tmp, k_tmp, v_tmp = q, k.to(device=q.device, dtype=q.dtype) ,v.to(device=q.device, dtype=q.dtype)
        if self.qkv_layout == AttnQKVLayout.SBHD:
            q_tmp, k_tmp, v_tmp = q_tmp.transpose(0, 1), k_tmp.transpose(0, 1), v_tmp.transpose(0, 1)
        if self.qkv_layout != AttnQKVLayout.THD:
            result = self.process(chunk_q=q_tmp, chunk_k=k_tmp, chunk_v=v_tmp)
            if self.qkv_layout == AttnQKVLayout.SBHD:
                result = result.transpose(0, 1)
        else:
            # print(cu_seqlens_q)
            cu_seqlens_q = cu_seqlens_q.to(device=q.device)
            cu_seqlens_k = cu_seqlens_k.to(device=q.device)
            chunks_q = [q_tmp[cu_seqlens_q[i]:cu_seqlens_q[i+1]].unsqueeze(0) for i in range(cu_seqlens_q.shape[0] - 1)]
            chunks_k = [k_tmp[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0) for i in range(cu_seqlens_k.shape[0] - 1)]
            chunks_v = [v_tmp[cu_seqlens_k[i]:cu_seqlens_k[i+1]].unsqueeze(0) for i in range(cu_seqlens_k.shape[0] - 1)]
            processed_chunks = [self.process(chunk_q=chunk_q, chunk_k=chunk_k, chunk_v=chunk_v).squeeze(0) for chunk_q, chunk_k, chunk_v in zip(chunks_q, chunks_k, chunks_v)]
            result = torch.cat(processed_chunks, dim=0)
        return result

    
class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        self.sq = seqlen_q
        self.skv = seqlen_kv
        self.bq = block_size_q
        self.bkv = block_size_kv
        mask = torch.zeros((self.sq, self.skv))
        if self.window_size is not None or self.casual:
            max_size = max(self.sq, self.skv)
            pos_q = torch.arange(self.sq)
            pos_kv = torch.arange(self.skv)
            pos_q = pos_q + (max_size - self.sq)
            pos_kv = pos_kv + (max_size - self.skv)
            dist = pos_q.unsqueeze(-1) - pos_kv.unsqueeze(0)
            if self.window_size is not None:
                mask = mask.masked_fill(dist.abs() > self.window_size, float('-inf'))
            if self.casual:
                mask = mask.masked_fill(dist < 0, float('-inf'))
        total_sq = math.ceil(self.sq / self.bq) * self.bq
        total_skv = math.ceil(self.skv / self.bkv) * self.bkv
        self.mask = F.pad(input=mask, pad=(0, total_skv - self.skv, 0, total_sq - self.sq), mode="constant", value=-float('inf'))
    
    def help(self, chunk_q: torch.Tensor, chunk_k: torch.Tensor, chunk_v: torch.Tensor, block_idx_q: int, block_idx_kv: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bs_q, bq, nq, hd_q = chunk_q.shape
        bs_kv, bkv, nkv, hd_kv = chunk_k.shape
        assert(nq == self.nq and nkv == self.nkv)
        assert(bs_q == bs_kv)
        assert(self.hd == hd_q and self.hd == hd_kv)
        assert(self.bq == bq and self.bkv == bkv)
        if self.apply_qk_norm:
            # assert(self.nq != 0)
            # assert(self.nkv != 0)
            chunk_q = self.normq(chunk_q.view(bs_q, bq, self.nq * self.hd)).view(bs_q, bq, self.nq, self.hd)
            chunk_k = self.normk(chunk_k.view(bs_kv, bkv, self.nkv * self.hd)).view(bs_kv, bkv, self.nkv, self.hd)
        chunk_q = chunk_q.transpose(1, 2)
        chunk_k = chunk_k.transpose(1, 2)
        chunk_v = chunk_v.transpose(1, 2)
        key_states = torch.repeat_interleave(input=chunk_k, repeats=nq // nkv, dim=1)
        value_states = torch.repeat_interleave(input=chunk_v, repeats=nq // nkv, dim=1)
        if self.softmax_cap == None:
            attn_weights = (self.softmax_scale * (chunk_q @ key_states.transpose(2, 3))) / self.softmax_temp
        else:
            attn_weights = self.softmax_cap * torch.tanh(self.softmax_scale * chunk_q @ key_states.transpose(2, 3) / self.softmax_cap)
        mask = self.mask[block_idx_q * self.bq : (block_idx_q + 1) * self.bq, block_idx_kv * self.bkv : (block_idx_kv + 1) * self.bkv].to(device=chunk_q.device, dtype=chunk_q.dtype)
        # print(mask)
        attn_weights = attn_weights + mask
        # print(attn_weights)
        lse_local = torch.logsumexp(attn_weights, dim=-1, keepdim=False)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        # print(attn_weights)
        # torch.manual_seed(self.softmax_dropout_seed)
        # l, r = self.softmax_clip_range
        # attn_weights = self.dropout(torch.clip((r - l) * attn_weights + l, 0, 1))
        result = attn_weights @ value_states
        # print(result.shape)
        return result.transpose(1, 2), lse_local

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        q_tmp = q
        k_tmp = k.to(device=q.device, dtype=q.dtype)
        v_tmp = v.to(device=q.device, dtype=q.dtype)
        local_o, local_lse = self.help(q_tmp, k_tmp, v_tmp, block_idx_q, block_idx_kv)
        sq_start = block_idx_q * self.bq
        sq_end = min(self.sq, (block_idx_q + 1) * self.bq)
        # print(local_o.dtype)
        # print(global_o.dtype)
        # print(local_lse.dtype)
        # print(global_lse.dtype)
        lse_1 = local_lse[:,:,0:(sq_end - sq_start)]
        # print(lse_1)
        # print(global_lse)
        lse_2 = global_lse[:,:,sq_start:sq_end]
        lse_min = torch.minimum(lse_1, lse_2)
        lse_max = torch.maximum(lse_1, lse_2)
        # print(lse_1)
        # print(lse_2)
        # print(lse_min)
        # print(lse_max)
        lse_tmp = lse_min - lse_max
        lse_tmp.nan_to_num_(nan=0.0)
        lse = lse_max + F.softplus(lse_tmp)
        # print(global_lse)
        # print(lse_1)
        # print(lse_2)
        # print(lse)
        c1 = torch.exp((lse_1 - lse).nan_to_num(nan=0.0))
        c2 = torch.exp((lse_2 - lse).nan_to_num(nan=0.0))
        # lse.nan_to_num_(nan=0.0)
        # c1 = torch.exp(lse_1 - lse)
        # c2 = torch.exp(lse_2 - lse)
        # print(c1)
        # print(c2)
        # print(global_o)
        local_o = local_o.transpose(1,2)
        global_o.transpose_(1, 2)
        # print(global_o[:,:,sq_start:sq_end,:])
        global_o[:,:,sq_start:sq_end,:] = global_o[:,:,sq_start:sq_end,:] * c2.unsqueeze(-1) + local_o[:,:,0:(sq_end - sq_start),:] * c1.unsqueeze(-1)
        # print(global_o)
        global_o.transpose_(1, 2)
        # global_o.nan_to_num_(nan=0.0)
        # print(global_o)
        global_lse[:,:,sq_start:sq_end] = lse
        # print(global_lse)
        # print(global_o)