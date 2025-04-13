from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """
    batch_size, seq_len, hidden_size = input.shape
    embed_size = weight.shape[1]
    if top_k == None:
        top_k = seq_len
    top_k_indices = torch.topk(probs, top_k, dim=-1).indices
    mask = torch.zeros_like(probs, dtype=bool).to(device=input.device)
    mask.scatter_(1, top_k_indices, True)
    mask = mask & (probs >= top_p)
    if grad_output == None:
        with torch.no_grad():
            output_tensor_1 = input[mask]
            reshaped_input = output_tensor_1.view(-1, num_heads, hidden_size // num_heads)
            reshaped_weight = weight.view(num_heads, hidden_size // num_heads, embed_size)
            # print(reshaped_input.shape)
            # print(reshaped_weight.shape)
            # output_tensor_2 = torch.empty(reshaped_input.shape[0], num_heads, embed_size).to(device= input.device, dtype=input.dtype)
            # for i in range(output_tensor_2.shape[0]):
            #     for j in range(output_tensor_2.shape[1]):
            #         output_tensor_2[i][j] = reshaped_input[i][j] @ reshaped_weight[j]
            output_tensor_2 = torch.einsum('bij,ijk->bik', reshaped_input, reshaped_weight)
            # print(output_tensor_2.shape)
            return output_tensor_2, None, None 
    else:
        # assert(input.requires_grad == False)
        # assert(input.grad == None)
        input.requires_grad_()
        weight.requires_grad_()
        output_tensor_1 = input[mask]
        reshaped_input = output_tensor_1.view(-1, num_heads, hidden_size // num_heads)
        reshaped_weight = weight.view(num_heads, hidden_size // num_heads, embed_size)
        # output_tensor_2 = torch.empty(reshaped_input.shape[0], num_heads, embed_size, requires_grad=True).to(device= input.device, dtype=input.dtype)
        # for i in range(output_tensor_2.shape[0]):
        #     for j in range(output_tensor_2.shape[1]):
        #         output_tensor_2[i][j] = reshaped_input[i][j] @ reshaped_weight[j]
        output_tensor_2 = torch.einsum('bij,ijk->bik', reshaped_input, reshaped_weight)
        output_tensor_2.backward(grad_output)
        dinput = input.grad.clone()
        dweight = weight.grad.clone()
        input.grad = None
        weight.grad = None
        input.requires_grad_(False)
        weight.requires_grad_(False)
        # assert(input.grad == None)
        # assert(input.requires_grad == False)
        return output_tensor_2, dinput, dweight



def apply_rotary_pos_emb(
    input: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
) -> torch.Tensor:
    """Applies rotary positional embedding to the input tensor.
    
    Args:
        input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        cos(torch.Tensor): cos basis tensor, with shape: [seq_len, head_dim]
        sin(torch.Tensor): sin basis tensor, with shape: [seq_len, head_dim]
    
    Returns:
        output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
    """
    # raise NotImplementedError("TODO: Assignment1 - Task3")
    x = input.transpose(1, 2).to(device=cos.device, dtype=torch.float32)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x_rotate_half = torch.cat((-x2, x1), dim=-1)
    x_embed = (x * cos.to(dtype=torch.float32)) + (x_rotate_half * sin.to(dtype=torch.float32))
    return x_embed.transpose(1,2).to(device=input.device, dtype=input.dtype)