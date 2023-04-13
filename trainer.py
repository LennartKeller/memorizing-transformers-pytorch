import logging
from math import ceil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, BatchEncoding
from datasets import Dataset

logger = logging.getLogger(__name__)

@dataclass
class MemorizingTransformerTrainingArguments(TrainingArguments):
    ...
    # max_train_segments: Union[int, None] = field(
    #     default=None,
    #     metadata={
    #         "help": "Maxmimum of segments per text while training. If None all segments are used."
    #     }

    # )
    # max_seq_length: Optional[int] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "The maximum total input sequence length after tokenization. Sequences longer "
    #             "than this will be chunked into segments."
    #         )
    #     },
    # )

class MemorizingTransformerTrainer(Trainer):
    # def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    #     segments = self._split_batch_into_segments(inputs=inputs)
    #     segment_outputs = []
    #     if self.args.max_train_segments is not None:
    #         segments = segments[:self.args.max_train_segments]
    #     for segment in segments:
    #         segment_output = super().training_step(model, segment)
    #         segment_outputs.append(segment_output)
    #     outputs = self._gather_train_loss(segment_outputs)
    #     return outputs
    
    # def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     segments = self._split_batch_into_segments(inputs=inputs)
    #     segment_outputs = []
    #     for segment in segments:
    #         segment_output = super().prediction_step(model, segment, prediction_loss_only, ignore_keys)
    #         segment_outputs.append(segment_output)
    #     outputs = self._gather_prediction_outputs(segment_outputs)
    #     return outputs
    
    # @staticmethod
    # def _gather_train_loss(segment_outputs: List[torch.Tensor]) -> torch.Tensor:
    #     return sum(segment_outputs) / len(segment_outputs)

    # @staticmethod
    # def _gather_prediction_outputs(segment_outputs: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    #     all_losses, all_logits, all_labels = [], [], []
    #     for loss, logits, labels in segment_outputs:
    #         all_losses.append(loss)
    #         all_logits.append(logits)
    #         all_labels.append(labels)
    #     gathered_loss = sum(all_losses) / len(all_losses)
    #     # Concat along sequence dimension
    #     gathered_logits = torch.cat(all_logits, dim=1) if all_logits[0] is not None else None
    #     gathered_labels = torch.cat(all_labels, dim=1) if all_labels[0] is not None else None

    #     return gathered_loss, gathered_logits, gathered_labels
    
    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     """
    #     Disable the built-in train sampler, to perform our own
    #     presumably much more inefficient sorting of the train dataset.
    #     """
    #     return None

    def get_train_dataloader(self) -> DataLoader:
        # TODO the original version of this methods bugs for some reasons.
        # Check if or live with it.
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        # TODO here is somethig broken too....
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=None,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

class RememBertTrainer(Trainer):
    ...
