import torch
from collections import defaultdict
from typing import Dict, List, Tuple

from transformers import BatchEncoding
from transformers.modeling_outputs import ModelOutput

class LongDocumentModel:
    def __call__(self, cls):
        setattr(cls, "_split_batch_into_segments", self._split_batch_into_segments)
        setattr(cls, "_gather_results_from_dicts", self._gather_results_from_dicts)
        setattr(cls, "_gather_results_from_tuples", self._gather_results_from_tuples)
    
    def _split_batch_into_segments(self, inputs: BatchEncoding) -> List[BatchEncoding]:
        max_seq_len = self.config.max_seq_len
        chunked_data = defaultdict(list)
        for key, tensor in inputs.items():
            splitted_tensors = tensor.split(max_seq_len, dim=1)
            chunked_data[key].extend(splitted_tensors)
        segments = [
            BatchEncoding({key: chunks[i] for key, chunks in chunked_data.items()})
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
            (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, sum(tensors) / len(tensors))
            # This version only keeps the last loss to pass it to trainer because the others were already backpropagated.
            # (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, tensors[-1] / len(all_outputs))
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
            # This version only keeps the last loss to pass it to trainer because the others were already backpropated.
            # loss = all_losses[-1] / len(gathered_outputs)
            loss = sum(all_losses) / len(all_losses)
        outputs = tuple(torch.cat(entries, dim=1) for entries in gathered_outputs)
        if labels is not None:
            outputs = (loss,) + outputs
        return outputs

if __name__ == "__main__":
        
    @LongDocumentModel
    class A:
        ...
    
    a = A()
    a._split_batch_into_segments