from inspect import signature
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from memorizing_transformers_pytorch.memorizing_transformers_encoder_pytorch import MemorizingTransformerEncoder

class HFMemorizingTransformerConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class HFMemorizingTransformerModel(PreTrainedModel):
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

class HFMemorizingTransformerForMaskedLM(HFMemorizingTransformerModel):

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
        
