from inspect import signature
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from memorizing_transformers_pytorch.memorizing_transformers_encoder_pytorch import MemorizingTransformerEncoder
from memorizing_transformers_pytorch.knn_memory import DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY

class MemorizingTransformerConfig(PretrainedConfig):
    def __init__(
            self,
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
    
    def forward(self, input_ids: Tensor, return_dict: bool = True, *args, **kwargs) -> BaseModelOutput:
        batch_size, *_ = input_ids.size()
        knn_memories = self.encoder.create_knn_memories(batch_size=batch_size) 
        embeddings = self.encoder(input_ids, knn_memories = knn_memories)
        
        if return_dict:
            outputs = BaseModelOutput(last_hidden_state=embeddings)
        else:
            outputs = (embeddings,)
        
        return outputs
    
    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.token_emb

class MemorizingTransformerForMaskedLM(MemorizingTransformerModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.classifier = nn.Linear(self.encoder.dim, self.encoder.num_tokens)

    def forward(self, input_ids: Tensor, labels: Tensor = None, return_dict: bool = True, *args, **kwargs):
        batch_size, *_ = input_ids.size()
        knn_memories = self.encoder.create_knn_memories(batch_size=batch_size) 
        embeddings = self.encoder(input_ids, knn_memories = knn_memories)
        logits = self.classifier(embeddings)
        
        if labels is not None:
            *_, num_tokens = logits.size()
            loss = F.cross_entropy(logits.view(-1, num_tokens), labels.view(-1))
        
        if return_dict:
            outputs = MaskedLMOutput(logits=logits)
            if labels is not None:
                outputs["loss"] = loss
        else:
            outputs = (embeddings,)
            if labels is not None:
                outputs += (loss,)
        
        return outputs
        
