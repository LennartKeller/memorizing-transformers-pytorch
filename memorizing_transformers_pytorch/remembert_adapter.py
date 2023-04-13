import torch
from torch import Tensor, nn
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from transformers import BatchEncoding
from transformers.modeling_outputs import ModelOutput


class RememBertLongDocumentWrapper(nn.Module):
    def __init__(self, module, max_seq_len=512, max_train_segments=5) -> None:
        super().__init__()
        self._module = module
        self.max_seq_len = max_seq_len
        self.max_train_segments = max_train_segments
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, return_dict: bool = True, *args, **kwargs):
        
        if labels is not None:
            inputs = BatchEncoding(dict(input_ids=input_ids, labels=labels, **kwargs))
        else:
            inputs = BatchEncoding(dict(input_ids=input_ids, **kwargs))

        segments = self._split_batch_into_segments(inputs)
        if self.training and self.max_train_segments:
            segments = segments[:self.max_train_segments]
    
        batch_size = inputs["input_ids"].size(0)
        outputs = []
        with self._module.knn_memories_context(batch_size=batch_size) as knn_memories:
            for segment in segments:
                segment["knn_memories"] = knn_memories
                segment_outputs = self._module(return_dict=return_dict, **segment)
                outputs.append(segment_outputs)
        if return_dict:
            concatenated_outputs = self._gather_results_from_dicts(outputs)
        else:
            concatenated_outputs = self._gather_results_from_tuples(outputs)
        return concatenated_outputs
    
    def _split_batch_into_segments(self, inputs: BatchEncoding) -> List[BatchEncoding]:
        max_seq_len = self.max_seq_len
        chunked_data = defaultdict(list)
        for key, tensor in inputs.items():
            splitted_tensor = tensor.split(max_seq_len, dim=1)
            chunked_data[key].extend(splitted_tensor)
        segments = [
            BatchEncoding({key: chunks[i].contiguous() for key, chunks in chunked_data.items()})
            for i in range(len(chunked_data["input_ids"]))
        ]
        return segments
    
    @staticmethod
    def _gather_results_from_dicts(segment_outputs: List[ModelOutput]) -> Dict[str, torch.Tensor]:
        leader_output = segment_outputs[0]        
        all_outputs = defaultdict(list)
        for key in leader_output.keys():
            for segment_output in segment_outputs:
                all_outputs[key].append(segment_output[key])
        
        concatenated_outputs = dict(
            (key, torch.cat(tensors, dim=1).contiguous()) if key != "loss" else (key, sum(tensors) / len(tensors))
            for key, tensors in all_outputs.items()
        )
        return concatenated_outputs
    
    @staticmethod
    def _gather_results_from_tuples(segment_outputs: List[Tuple[torch.Tensor]], labels: torch.Tensor = None) -> Tuple[torch.Tensor]:
        # Tuple order ([loss], <logits/embs>, <rest>)
        leader_output = segment_outputs[0]
        gathered_outputs = [[] for _ in range(len(leader_output))]
        for segment_output in segment_outputs:
            for idx, entry in enumerate(segment_output):
                gathered_outputs[idx].append(entry)
        if labels is not None:
            all_losses = gathered_outputs.pop(0)
            loss = sum(all_losses) / len(all_losses)
        outputs = tuple(torch.cat(entries, dim=1).contiguous() for entries in gathered_outputs)
        if labels is not None:
            outputs = (loss,) + outputs
        return outputs
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def __setattr__(self, name, value):
        if name == '_module':
            super().__setattr__(name, value)
        else:
            setattr(self._module, name, value)

    def __delattr__(self, name):
        if name == '_module':
            super().__delattr__(name)
        else:
            delattr(self._module, name)

if __name__ == "__main__":
    ...
