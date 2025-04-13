from typing import Dict, Any, List, Union, Optional, Tuple, Sequence
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from ...utils import load_jsonl

from .base import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    BaseDatasetConfig,
    BaseDataset,
)


@config_dataclass
class QADatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Question-Answering Tasks"""
    
    question_key: str = make_fixed_field("question")
    answer_key: str = make_fixed_field("answer")
    
    question_prefix: str = make_fixed_field("QUESTION")
    answer_prefix: str = make_fixed_field("ANSWER")
    

class QADataset(BaseDataset):
    """Dataset Class for Question-Answering Tasks"""
    
    def __init__(
        self,
        config: QADatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize QADataset module
        Args:
            config (QADatasetConfig): qa dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        super().__init__()
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        if "qa" in data_files:
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
                    question = sample[self.config.question_key]
                    answer = sample[self.config.answer_key]
                    questions_id += self.tokenizer.encode(question + answer)
                    answers_id += self.tokenizer.encode(question + answer)
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
    
    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        return len(self.sample_list)
    
    def sample(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        return self.sample_list[idx]
    
    def num_batchs(self) -> int:
        """
        Returns the number of batchs in the dataset.
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        return len(self.batch_list)
    
    def batch(self, idx: int) -> Dict[str, Any]:
        """
        Returns a batch from the dataset.
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        return self.batch_list[idx]
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset, including the samples and batches.
            
        Args:
            seed (Optional[int], optional): Random seed. Defaults to None to be un-deterministic.
        """
        # raise NotImplementedError("TODO: Assignment5 - Task3")
        if seed is not None:
            torch.manual_seed(seed)
        random.shuffle(self.sample_list)
        random.shuffle(self.batch_list)
    