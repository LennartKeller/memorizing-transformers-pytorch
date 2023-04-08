"""
DEPRECATED!
TODOs:
    - Change chunking in favor from chunking from recurrent model.
    - Refactor adapter and model logic.
"""
from inspect import signature
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput
from memorizing_transformers_pytorch.memorizing_transformers_encoder_pytorch import MemorizingTransformerEncoder
from memorizing_transformers_pytorch.knn_memory import DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
from memorizing_transformers_pytorch.modeling_bert import BertModel, BertPreTrainedModel

class RememBertModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.encoder = BertModel(config=config)
    
    def forward(self, input_ids: Tensor, return_dict: bool = True, *args, **kwargs) -> BaseModelOutput:
        batch_size, *_ = input_ids.size()
        knn_memories = self.create_knn_memories(batch_size=batch_size) 
        outputs = self.encoder(input_ids, knn_memories = knn_memories)
        embeddings = outputs[0] if return_dict else outputs["last_hidden_state"]

        
        if return_dict:
            outputs = BaseModelOutput(last_hidden_state=embeddings)
        else:
            outputs = (embeddings,)
        
        return outputs


class RememBertForMaskedLM(RememBertModel):

    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = self.encoder
        del self.encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.vocab_size),
        )

    def forward(self, input_ids: Tensor, labels: Tensor = None, return_dict: bool = True, *args, **kwargs):
        batch_size, *_ = input_ids.size()
        knn_memories = self.create_knn_memories(batch_size=batch_size) 
        outputs = self.encoder(input_ids, knn_memories = knn_memories)
        embeddings = outputs[0] if return_dict else outputs["last_hidden_state"]
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
        
