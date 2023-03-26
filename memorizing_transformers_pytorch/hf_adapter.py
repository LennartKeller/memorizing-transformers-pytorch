from collections import defaultdict
import logging
from inspect import signature
from math import ceil
from typing import Dict, List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, ModelOutput
from memorizing_transformers_pytorch.memorizing_transformers_encoder_pytorch import MemorizingTransformerEncoder
from memorizing_transformers_pytorch.knn_memory import DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY

logger = logging.getLogger(__name__)

class MemorizingTransformerConfig(PretrainedConfig):
    def __init__(
            self,
            max_seq_len = 512,
            num_tokens = 20_000,
            dim = 768,
            depth = 12,
            dim_head = 64,
            heads = 12,
            knn_attn_heads = None,
            attn_dropout = 0.,
            intermediate_dim = 4096,
            ff_dropout = 0.,
            memorizing_layers = None,
            max_knn_memories = 250000,
            num_retrieved_memories = 32,
            clear_memories_on_sos_token_id = None,
            clear_memories_on_eos_token_id = None,
            knn_memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
            shift_knn_memories_down = 0.,
            pad_id = 0,
            xl_max_memories = 0,
            xl_memory_layers = None,
            shift_xl_memories_down = 0.,
            knn_memory_multiprocessing = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.knn_attn_heads = knn_attn_heads
        self.attn_dropout = attn_dropout
        self.intermediate_dim = intermediate_dim
        self.ff_dropout = ff_dropout
        self.memorizing_layers = memorizing_layers
        self.max_knn_memories = max_knn_memories
        self.num_retrieved_memories = num_retrieved_memories
        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id
        self.knn_memories_directory = knn_memories_directory
        self.shift_knn_memories_down = shift_knn_memories_down
        self.pad_id = pad_id
        self.xl_max_memories = xl_max_memories
        self.xl_memory_layers = xl_memory_layers
        self.shift_xl_memories_down = shift_xl_memories_down
        self.knn_memory_multiprocessing = knn_memory_multiprocessing



class MemorizingTransformerModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # If the config is read from disk, tuples become lists and are not hashable anymore...
        config = {arg: val if not isinstance(val, list) else tuple(val) for arg, val in config.to_dict().items()}
        
        self.encoder = MemorizingTransformerEncoder(**{
            arg: val
            for arg, val in config.items()
            if arg in signature(MemorizingTransformerEncoder.__init__).parameters
        })

    def forward(self, *args, **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        outputs = self._forward(*args, **kwargs)
        if isinstance(outputs, dict):
            return BaseModelOutput(**outputs)
        else:
            return outputs
    
    def forward_segment(
            self, 
            knn_memories,
            input_ids: torch.Tensor,
            return_dict: bool = True,
            *args,
            **kwargs
        ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:

        embeddings = self.encoder(input_ids, knn_memories = knn_memories)
        if return_dict:
            outputs = dict(last_hidden_state=embeddings)
        else:
            outputs = (embeddings,)
        return outputs
    
    def _forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        if labels is not None:
            inputs = BatchEncoding(dict(input_ids=input_ids, labels=labels, **kwargs))
        else:
            inputs = BatchEncoding(dict(input_ids=input_ids, **kwargs))
        segments = self._split_batch_into_segments(inputs)
        n_segments = len(segments) - 1
        print("ns", len(segments))
        segment_outputs = []
        batch_size, *_ = input_ids.size()
        # TODO backprop
        with self.encoder.knn_memories_context(batch_size=batch_size) as knn_memories:
            for seg_idx, segment in enumerate(segments):
                segment_output = self.forward_segment(knn_memories=knn_memories, return_dict=return_dict, **segment)
                # If model is in training mode, there are labels, and we process a non-final segment, then we do backprop.
                if self.training and labels is not None:
                    if seg_idx < n_segments:
                        segment_output = self._backpropagate_and_detach(
                            segment_output=segment_output,
                            return_dict=return_dict,
                            n_segments=n_segments
                        )
                segment_outputs.append(segment_output)
        if return_dict:
            outputs = self._gather_results_from_dicts(segment_outputs)
        else:
            outputs = self._gather_results_from_tuples(segment_outputs)

        return outputs
        
    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.token_emb
    
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
        text_length_rounded = self.config.max_seq_length * ceil(text_length / self.config.max_seq_length)
        n_segments = max(round(text_length_rounded // self.config.max_seq_length), 1)
        
        segments = [{} for _ in range(n_segments)]
        for key, tensor in inputs.items():
            segmented_tensors = tensor.chunk(n_segments, dim=-1) 
            for idx, (segment, segmented_tensor) in enumerate(zip(segments, segmented_tensors)):
                if idx == 0 and (seg_len := segmented_tensor.size(-1)) > self.config.max_seq_length:
                    logger.warning(f"Encountered segment with invalid length ({seg_len})")
                segment[key] = segmented_tensor
        return [BatchEncoding(segment) for segment in segments]

    @staticmethod
    def _gather_results_from_dicts(segment_outputs: List[ModelOutput]) -> Dict[str, torch.Tensor]:
        leader_output = segment_outputs[0]        
        all_outputs = defaultdict(list)
        for key in leader_output.keys():
            for segment_output in segment_outputs:
                all_outputs[key].append(segment_output[key])
        
        concatenated_outputs = dict(
            # (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, sum(tensors) / len(tensors))
            # This version only keeps the last loss to pass it to trainer because the others were already backpropated.
            (key, torch.cat(tensors, dim=1)) if key != "loss" else (key, tensors[-1] / len(all_outputs))
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
            loss = all_losses[-1] / len(gathered_outputs)
            # loss = sum(all_losses) / len(all_losses)
        outputs = tuple(torch.cat(entries, dim=1) for entries in gathered_outputs)
        if labels is None:
            outputs = (loss,) + outputs
        return outputs
    
    @staticmethod
    def _backpropagate_and_detach(segment_output, return_dict, n_segments):
        if return_dict:
            loss = segment_output["loss"]
        else:
            loss = segment_output[0]
        
        scaled_loss = loss / n_segments
        scaled_loss.backward()

        if return_dict:
            segment_output = {
                key: tensor.detach()
                for key, tensor in segment_output.items()
            }
        else:
            segment_output = tuple([tensor.detach() for tensor in segment_output])
        return segment_output


class MemorizingTransformerForMaskedLM(MemorizingTransformerModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.classifier = nn.Linear(self.encoder.dim, self.encoder.num_tokens)
    
    def forward(self, *args, **kwargs):
        outputs = self._forward(*args, **kwargs)
        if isinstance(outputs, dict):
            return MaskedLMOutput(**outputs)
        else:
            return outputs

    def forward_segment(self, knn_memories, input_ids: torch.Tensor, labels: torch.Tensor = None, return_dict: bool = True, *args, **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor]]:
        embeddings = self.encoder.forward(input_ids, knn_memories=knn_memories)
        logits = self.classifier(embeddings)
        
        if labels is not None:
            *_, num_tokens = logits.size()
            loss = F.cross_entropy(logits.reshape(-1, num_tokens), labels.reshape(-1))
        
        if return_dict:
            outputs = MaskedLMOutput(logits=logits)
            if labels is not None:
                outputs["loss"] = loss
        else:
            outputs = (embeddings,)
            if labels is not None:
                outputs += (loss,)
        
        return outputs
        
