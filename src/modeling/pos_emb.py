import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment1 - Task3")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.ratio = ratio
        self.dynamic = dynamic
        self.dtype = dtype
        self.device = device
        tmp = self.base * (self.ratio) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / (tmp ** (torch.arange(0, self.dim, 2).to(device=self.device, dtype=self.dtype) / self.dim))
        t = torch.arange(self.max_seq_len * self.ratio, device=self.device, dtype=self.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        # raise NotImplementedError("TODO: Assignment1 - Task3")
        seq_len = input.shape[1]
        if seq_len + offset > self.max_seq_len * self.ratio:
            ratio = (seq_len + offset + self.max_seq_len - 1) // self.max_seq_len
            if ratio % 2 == 1:
                ratio += 1
            base = self.base * (ratio) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).to(device=self.device, dtype=self.dtype) / self.dim))
            t = torch.arange(self.max_seq_len * ratio, device=self.device, dtype=self.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
            if self.dynamic:
                self.ratio = ratio
                self.register_buffer("cos_cached", cos, persistent=False)
                self.register_buffer("sin_cached", sin, persistent=False)
            return apply_rotary_pos_emb(input, cos[offset : seq_len + offset], sin[offset : seq_len + offset])
        else:
            return apply_rotary_pos_emb(input, self.cos_cached[offset : seq_len + offset], self.sin_cached[offset : seq_len + offset])