from typing import Dict, Any, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling.prompt import PaddingSide, TruncateSide
from src.utils import load_jsonl

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from .base import BaseDatasetConfig

from .qa import QADataset


@config_dataclass
class ChatDatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Chatbot Tasks"""
    
    conversations_key: str = make_fixed_field("conversations")
    role_key: str = make_fixed_field("role")
    content_key: str = make_fixed_field("content")
    
    user_role_value: str = make_fixed_field("user")
    bot_role_value: str = make_fixed_field("chatbot")
    
    user_role_prefix: str = make_fixed_field("USER")
    bot_role_prefix: str = make_fixed_field("CHATBOT")


class ChatDataset(QADataset):
    """Dataset Class for Chatbot Tasks"""
    
    def __init__(
        self,
        config: ChatDatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize ChatDataset module
        Args:
            config (ChatDatasetConfig): chat dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        super().__init__(config=config, tokenizer=tokenizer, data_files=data_files)
        self.config = config
        self.tokenizer = tokenizer
        self.sample_list = load_jsonl(data_files)
        self.batch_list = []
        for i in range(0, len(self.sample_list), self.config.batch_size):
            batch_dict = {}
            if self.config.drop_last_incomplete_batch and i + self.config.batch_size > len(self.sample_list):
                pass
            else:
                end = min(len(self.sample_list), i + self.config.batch_size)
            batch_dict[self.config.samples_key] = self.sample_list[i:end]
            batch_dict[self.config.cu_seqlens_key] = None
            questions_id = []
            answers_id = []
            for sample in self.sample_list[i:end]:
                user = ""
                chatbot = ""
                for sentence in sample[self.config.conversations_key]:
                    user += sentence[self.config.content_key]
                questions_id += self.tokenizer.encode(user)
                answers_id += self.tokenizer.encode(user)
            for i in range(len(questions_id)):
                sq = questions_id[i].shape[0]
                if sq > self.config.seq_len:
                    if self.config.truncate_side == TruncateSide.LEFT:
                        questions_id[i] = questions_id[i][sq - self.config.seq_len].unsqueeze(0)
                    else:
                        questions_id[i] = questions_id[i][0:self.config.seq_len].unsqueeze(0)
                else:
                    if self.config.padding_side == PaddingSide.LEFT:
                        questions_id[i] = F.pad(questions_id[i], (self.config.seq_len - sq, 0), "constant", self.tokenizer.bos_id).unsqueeze(0)
                    else:
                        questions_id[i] = F.pad(questions_id[i], (0, self.config.seq_len - sq), "constant", self.tokenizer.eos_id).unsqueeze(0)
                sq = answers_id[i].shape[0]
                if sq > self.config.seq_len:
                    if self.config.truncate_side == TruncateSide.LEFT:
                        answers_id[i] = answers_id[i][sq - self.config.seq_len].unsqueeze(0)
                    else:
                        answers_id[i] = answers_id[i][0:self.config.seq_len].unsqueeze(0)
                else:
                    if self.config.padding_side == PaddingSide.LEFT:
                        answers_id[i] = F.pad(answers_id[i], (self.config.seq_len - sq, 0), "constant", self.config.ignore_idx).unsqueeze(0)
                    else:
                        answers_id[i] = F.pad(answers_id[i], (0, self.config.seq_len - sq), "constant", self.config.ignore_idx).unsqueeze(0)
            input_ids = torch.cat(questions_id, dim=0)
            labels = torch.cat(answers_id, dim=0)
            batch_dict[self.config.input_ids_key] = input_ids
            batch_dict[self.config.labels_key] = labels
            self.batch_list.append(batch_dict)
    