import re
from typing import Tuple
import torch
from torch import Tensor
from memorizing_transformers_pytorch import MemorizingTransformerModel, MemorizingTransformerConfig
from transformers import AutoModel, AutoTokenizer


def print_sizes(state_dict):
    for name, tensor in state_dict.items():
        print(name, tensor.size())

def bert_join_key_value_projections(src_match: re.Match, trgt_fstring: str, weights: Tensor) -> Tuple[str, Tensor]:
    ...

# HParams of Bert that can be directly translated to MemeTRF-HParams.
BERT_CONFIG_TRANSLATE_MAP = {
    "num_tokens": "vocab_size",
    "num_hidden_layers": "depth",
    "num_attention_heads": "heads",
    "hidden_dropout_prob": "ff_dropout",
}
# Rules to convert Bert weights to corresponding memory model weights.
BERT_WEIGHT_CONVERSION_MAP = {
    # Token embeddings
    "embeddings.word_embeddings.weight": "encoder.token_emb.weight",
    # AttnLayer Query Projection
}

if __name__ == "__main__":
    
   bert_model = AutoModel.from_pretrained("deepset/gbert-large") 
   bert_tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
   print_sizes(bert_model.state_dict())



