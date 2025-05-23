import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupRMSNorm(nn.Module):
    """Group RMS Norm module
    This is a variant of RMS Norm that \
        evenly splits the hidden dimension into groups, and \
        applies root-mean-square normalization with \
            learnable scaling transformation on each i-th group individually.
    """
    
    def __init__(self, 
        hidden_size: int, 
        group_size: int,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Group RMS Norm module
        
        Args:
            hidden_size(int): hidden dimension size
            group_size(int): group size
            eps(float, default = 1e-5): epsilon
            init_range(tuple, default = (-1.0, 1.0)): the range of the uniform distribution to initialize learnable scaling parameters
            init_seed(int, default = 42): seed for the initialization
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment1 - Task0")
        self.hidden_size = hidden_size
        if group_size == None:
            group_size = hidden_size
        self.group_size = group_size
        self.group_num = hidden_size // group_size
        self.eps = eps
        self.init_range = init_range
        self.init_seed = init_seed
        self.dtype = dtype
        self.device = device
        self.reset_parameters()
        
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """The forward pass for Group RMS Norm module

        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, hidden_size]
            
        Returns:
            output(torch.Tensor): normalized output tensor, with shape: [batch_size, seq_len, hidden_size]
        """
        # raise NotImplementedError("TODO: Assignment1 - Task0")
        bs, sq, hd = input.shape
        input_dtype = input.dtype
        input_reshape = input.view(bs, sq, self.group_num, self.group_size).to(dtype=torch.float32)
        weight = self.weight.view(self.group_num, self.group_size).to(device=input.device, dtype=torch.float32)
        variance = input_reshape.pow(2).mean(-1, keepdim=True)
        output_1 = input_reshape * torch.rsqrt(variance + self.eps)
        output = (weight * output_1).view(bs, sq, hd).to(dtype=input_dtype)
        return output

    
    def reset_parameters(self) -> None:
        """Initialize learnable scaling parameters for Group RMS Norm from a uniform distribution"""
        # raise NotImplementedError("TODO: Assignment1 - Task0")
        torch.manual_seed(self.init_seed)
        self.weight = nn.Parameter(torch.ones(self.hidden_size, device=self.device, dtype=self.dtype).uniform_(*self.init_range))
        # self.weight = torch.ones(self.hidden_size, device=self.device, dtype=self.dtype).uniform_(*self.init_range)

