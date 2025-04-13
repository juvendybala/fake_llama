from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class MLPActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    BILINEAR = "bilinear"


class DenseMLPWithLoRA(nn.Module):
    """Dense MLP module with LoRA adapters
    This is a GLU-style dense MLP layer with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Dense MLP module with LoRA adapters
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = "silu"): activation type
            init_base_seed(int, default = 42): seed for base weight initialization
            lora_rank(int, default = 0): lora rank, if 0, then no lora to apply
            lora_alpha(Optional[float], default = None): lora alpha, if None, then set to lora_rank
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("Assignment2 - Task1")
        self.hidden_size = hidden_size
        self.ffh_size = ffh_size
        self.activation_type = activation_type
        self.init_base_seed = init_base_seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_rate = lora_dropout_rate
        self.lora_dropout_seed = lora_dropout_seed
        self.lora_init_base_seed = lora_init_base_seed
        self.dtype = dtype
        self.device = device
        if self.lora_alpha == None:
            self.lora_alpha = self.lora_rank
        self.up = nn.Parameter(torch.empty((self.hidden_size, self.ffh_size), requires_grad=True, device=self.device, dtype=self.dtype))
        self.gate = nn.Parameter(torch.empty((self.hidden_size, self.ffh_size), requires_grad=True, device=self.device, dtype=self.dtype))
        self.down = nn.Parameter(torch.empty((self.ffh_size, self.hidden_size), requires_grad=True, device=self.device, dtype=self.dtype))
        if self.lora_rank != 0:
            self.A_r = nn.Parameter(torch.empty((self.hidden_size, self.lora_rank), requires_grad=True, device=self.device, dtype=self.dtype))
            self.B_r = nn.Parameter(torch.empty((self.lora_rank, self.hidden_size), requires_grad=True, device=self.device, dtype=self.dtype))
            torch.manual_seed(self.lora_dropout_seed)
            self.dropout = nn.Dropout(self.lora_dropout_rate)
        self.reset_parameters()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Dense MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("Assignment2 - Task1")
        X = input.to(dtype=self.dtype, device=self.device)
        if self.activation_type == MLPActivationType.SILU:
            o1 = (F.silu(X @ self.gate) * (X @ self.up)) @ self.down
        elif self.activation_type == MLPActivationType.RELU:
            o1 = (F.relu(X @ self.gate) * (X @ self.up)) @ self.down
        elif self.activation_type == MLPActivationType.GELU:
            o1 = (F.gelu(X @ self.gate) * (X @ self.up)) @ self.down
        elif self.activation_type == MLPActivationType.SIGMOID:
            o1 = (F.sigmoid(X @ self.gate) * (X @ self.up)) @ self.down
        else:
            o1 = ((X @ self.gate) * (X @ self.up)) @ self.down
        if self.lora_rank != 0:
            torch.manual_seed(self.lora_dropout_seed)
            o2 = self.dropout(self.lora_alpha / (self.lora_rank * 1.0) * (X @ self.A_r @ self.B_r))
            o1 = o1 + o2
        return o1.to(device=input.device, dtype=input.dtype)
    
    def reset_parameters(self):
        """Initialize the weights of the Dense MLP module with LoRA adapters
        from a normal distribution (or a uniform distribution for lora weights)
        """
        # raise NotImplementedError("Assignment2 - Task1")
        if self.activation_type == MLPActivationType.BILINEAR or self.activation_type == MLPActivationType.SIGMOID:
            torch.manual_seed(self.init_base_seed + 1)
            nn.init.xavier_normal_(self.up.T)
            torch.manual_seed(self.init_base_seed + 2)
            nn.init.xavier_normal_(self.gate.T)
            torch.manual_seed(self.init_base_seed + 3)
            nn.init.xavier_normal_(self.down.T)
            if self.lora_rank != 0:
                torch.manual_seed(self.lora_init_base_seed + 1)
                nn.init.xavier_uniform_(self.A_r)
                torch.manual_seed(self.lora_init_base_seed + 2)
                nn.init.xavier_uniform_(self.B_r)
        else:
            torch.manual_seed(self.init_base_seed + 1)
            nn.init.kaiming_normal_(self.up.T, mode="fan_in", nonlinearity="relu")
            torch.manual_seed(self.init_base_seed + 2)
            nn.init.kaiming_normal_(self.gate.T, mode="fan_in", nonlinearity="relu")
            torch.manual_seed(self.init_base_seed + 3)
            nn.init.kaiming_normal_(self.down.T, mode="fan_in", nonlinearity="relu")
            if self.lora_rank != 0:
                torch.manual_seed(self.lora_init_base_seed + 1)
                nn.init.kaiming_uniform_(self.A_r.T, mode="fan_in", nonlinearity="relu")
                torch.manual_seed(self.lora_init_base_seed + 2)
                nn.init.kaiming_uniform_(self.B_r.T, mode="fan_in", nonlinearity="relu")

    
class SparseMLPWithLoRA(nn.Module):
    """Sparse MLP module with LoRA adapters
    This is a GLU-style sparse MLP layer with LoRA adapters, \
        where the sparcity is implemented as Mixture of Experts (MoE), \
            and each expert is a dense MLP with LoRA adapters.
    """
    
    def __init__(self,
        hidden_size: int,
        ffh_size: int,
        activation_type: MLPActivationType = MLPActivationType.SILU,
        num_experts: int = 1,
        moe_topk: int = 1,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        lora_rank: int = 0,
        lora_alpha: Optional[float] = None,
        lora_dropout_rate: float = 0.0,
        lora_dropout_seed: int = 42,
        lora_init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Sparse MLP module with LoRA adapters
        
        Args:
            hidden_size(int): hidden dimension size
            ffh_size(int): hidden dimension size
            activation_type(MLPActivationType, default = MLPActivationType.SILU): activation type
            num_experts(int, default = 1): number of (global) experts, which can deduce expert_size = ffh_size // num_experts
            moe_topk(int, default = 1): topk-routing for MoE to control the sparcity
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean for the initialization
            init_std(float, default = 1.0): std for the initialization
            init_base_seed(int, default = 42): seed for the initialization
            lora_rank(int, default = 0): lora rank
            lora_alpha(Optional[float], default = None): lora alpha
            lora_dropout_rate(float, default = 0.0): lora dropout rate
            lora_dropout_seed(int, default = 42): lora dropout seed
            lora_init_base_seed(int, default = 42): seed for lora weight initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("Assignment2 - Task2")
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.e = ffh_size // num_experts
        self.moe_topk = moe_topk
        self.rank = rank
        self.nle = num_experts // world_size
        self.init_mean = init_mean
        self.init_std = init_std
        self.dtype = dtype
        self.device = device
        self.init_base_seed = init_base_seed
        self.model_list = nn.ModuleList([DenseMLPWithLoRA(hidden_size=self.hidden_size, ffh_size=self.e, activation_type=activation_type, init_base_seed=self.init_base_seed + self.rank *self.nle + idx,
                                            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout_rate=lora_dropout_rate, lora_dropout_seed=lora_dropout_seed + self.rank * self.nle + idx,lora_init_base_seed=lora_init_base_seed + 
                                            self.rank * self.nle + idx,dtype=self.dtype, device=self.device) for idx in range(self.nle)])
        self.init = False
        self.reset_parameters()
        self.init = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Sparse MLP module with LoRA adapters
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("Assignment2 - Task2")
        batch_size, sequence_length, hidden_dim = input.shape
        hidden_states = input.view(-1, hidden_dim).to(dtype=torch.float32, device=self.device)
        router_logits = hidden_states @ self.G
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.moe_topk, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        hidden_states = hidden_states.to(dtype=self.dtype)
        routing_weights = routing_weights.to(dtype=self.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=self.dtype, device=self.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.nle):
            expert_layer = self.model_list[expert_idx]
            # print(expert_mask[self.rank * self.nle + expert_idx])
            idx, top_x = torch.where(expert_mask[self.rank * self.nle + expert_idx])
            indices = torch.argsort(top_x)
            idx = idx[indices]
            top_x = top_x[indices]
            # print(idx)
            # print(top_x)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states.to(device=input.device, dtype=input.dtype)

        
    def reset_parameters(self):
        """Initialize the weights of each local expert from its own distribution \
            and the gating layer from a normal distribution
        """
        # raise NotImplementedError("Assignment2 - Task2")
        torch.manual_seed(self.init_base_seed)
        self.G = nn.Parameter(torch.normal(mean=self.init_mean, std=self.init_std,size=(self.hidden_size, self.num_experts), dtype=torch.float32, device=self.device))
        if self.init:
            for model in self.model_list:
                model.reset_parameters()