import re
from typing import Dict, Optional
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import find_format_keys


class BatchLayout(Enum):
    """Batch Layout Enum"""
    CONCAT = "concat"
    STACK = "stack"


class PaddingSide(Enum):
    """Padding Side Enum"""
    LEFT = "left"
    RIGHT = "right"


class TruncateSide(Enum):
    """Truncate Side Enum"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PromptType(Enum):
    """Prompt Types Enum"""
    
    SYSTEM = "system"
    CONTEXT = "context"
    QUERY = "query"
    RESPONSE = "response"
    PROMPT = "prompt" # NOTE: prompt = system + context + query
    ALL = "all" # NOTE: all = prompt + response
    

class PromptTemplate(nn.Module):
    """Prompt Template module"""
    
    def __init__(self, template_str: str = ""):
        """Initialize Prompt Template module
        
        Args:
            template_str (str): the template string with the format: "....{key1}...{key2}..."
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment5 - Task2")
        self.template_str = template_str
        self.keys_with_defaults = {key: None for key in re.findall(r'\{(\w+)\}', template_str)}
    
    def keys(self) -> Dict[str, Optional[str]]:
        """Get the keys with its default values of the prompt template as a dictionary
        NOTE: if any key has not been set with default value, then use `None` as a placeholder
        """
        # raise NotImplementedError("TODO: Assignment5 - Task2")
        return self.keys_with_defaults
    
    def set_default(self, **kwargs: Optional[Dict[str, str]]) -> None:
        """Set the default values of the prompt template keys"""
        # raise NotImplementedError("TODO: Assignment5 - Task2")
        for key, value in kwargs.items():
            if key in self.keys_with_defaults:
                self.keys_with_defaults[key] = value
    
    def forward(self, **kwargs: Optional[Dict[str, str]]) -> str:
        """Set the prompt template keys with the given keyword argument to get the formatted prompt
        NOTE:
            1. if certain prompt template key has not been set with its default value, then its corresponding kwarg should be provided
            2. if certain key in the kwargs is not found in the keys of the prompt template, just ignore it
        """
        # raise NotImplementedError("TODO: Assignment5 - Task2")
        for key in self.keys_with_defaults:
            if self.keys_with_defaults[key] is None and key not in kwargs:
                raise ValueError
        all_kwargs = self.keys_with_defaults.copy()
        all_kwargs.update(kwargs)
        return self.template_str.format(**all_kwargs)

