import logging
from math import ceil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import Trainer, TrainingArguments, BatchEncoding

logger = logging.getLogger(__name__)

@dataclass
class MemorizingTransformerTrainingArguments(TrainingArguments):
    max_train_segments: Union[int, None] = field(
        default=None,
        metadata={
            "help": "Maxmimum of segments per text while training. If None all segments are used."
        }

    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be chunked into segments."
            )
        },
    )

class MemorizingTransformerTrainer(Trainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        segments = self._split_batch_into_segments(inputs=inputs)
        segment_outputs = []
        if self.args.max_train_segments is not None:
            segments = segments[:self.args.max_train_segments]
        for segment in segments:
            segment_output = super().training_step(model, segment)
            segment_outputs.append(segment_output)
        outputs = self._gather_train_loss(segment_outputs)
        return outputs
    
    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        segments = self._split_batch_into_segments(inputs=inputs)
        segment_outputs = []
        for segment in segments:
            segment_output = super().prediction_step(model, segment, prediction_loss_only, ignore_keys)
            segment_outputs.append(segment_output)
        outputs = self._gather_prediction_outputs(segment_outputs)
        return outputs
    
    def _split_batch_into_segments(self, inputs: BatchEncoding) -> List[BatchEncoding]:
        """
        Training a memorizing transformer requires a special batching of texts,
        because it is crucial to process longer texts segmentwise 
        in order to train or - during inference - leverage the memory.

        This function expects batches to be padded to max_length.

        It takes a batch of texts in (padded) original length and 
        splits them into round(batch_text_length // max_seq_length) segments.
        """
  
        
        text_length = inputs["input_ids"].size(1)
        # Round UP to next common denominator, to avoid overflowing segments...
        text_length_rounded = self.args.max_seq_length * ceil(text_length / self.args.max_seq_length)
        n_segments = max(round(text_length_rounded // self.args.max_seq_length), 1)
        
        segments = [{} for _ in range(n_segments)]
        for key, tensor in inputs.items():
            segmented_tensors = tensor.chunk(n_segments, dim=-1) 
            for idx, (segment, segmented_tensor) in enumerate(zip(segments, segmented_tensors)):
                if idx == 0 and (seg_len := segmented_tensor.size(-1)) > self.args.max_seq_length:
                    logger.warning(f"Encountered segment with invalid length ({seg_len})")
                segment[key] = segmented_tensor
        return [BatchEncoding(segment) for segment in segments]
    
    @staticmethod
    def _gather_train_loss(segment_outputs: List[torch.Tensor]) -> torch.Tensor:
        return sum(segment_outputs) / len(segment_outputs)

    @staticmethod
    def _gather_prediction_outputs(segment_outputs: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
        all_losses, all_logits, all_labels = [], [], []
        for loss, logits, labels in segment_outputs:
            all_losses.append(loss)
            all_logits.append(logits)
            all_labels.append(labels)
        gathered_loss = sum(all_losses) / len(all_losses)
        # Concat along sequence dimension
        gathered_logits = torch.cat(all_logits, dim=1) if all_logits[0] is not None else None
        gathered_labels = torch.cat(all_labels, dim=1) if all_labels[0] is not None else None

        return gathered_loss, gathered_logits, gathered_labels
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        Disable the built-in train sampler, to perform our own
        presumably much more inefficient sorting of the train dataset.
        """
        return None





if __name__ == "__main__":
    from transformers import HfArgumentParser
    args = HfArgumentParser((MemorizingTransformerTrainingArguments)).parse_args_into_dataclasses()
    print(args)