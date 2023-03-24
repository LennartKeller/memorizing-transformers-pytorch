import re
from collections import UserDict, OrderedDict
from typing import Callable, Dict, List, Tuple, Union
import torch
from torch import nn
from torch import Tensor
from memorizing_transformers_pytorch import MemorizingTransformerModel, MemorizingTransformerConfig
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

class ConversionMap(UserDict):
    def __init__(self, map):
        self.string_map = {}
        self.regex_map = {}
        for key, val in map.items():
            if isinstance(key, str):
                self.string_map[key] = val
            elif isinstance(key, re.Pattern):
                self.regex_map[key] = val
            else:
                raise TypeError(f"Encountered invalid key type ({type(key)}) in conversion map key {key}")
        
    def __getitem__(self, key):
        try:
            value = self.string_map[key]
            return value
        except KeyError:
            for pattern, value in self.regex_map.items():
                if (match := re.match(pattern, key)) is not None:
                    formatted_value = value.format(**match.groupdict())
                    return formatted_value
        raise KeyError(f"Key {key} not in ConversionMap")

def print_sizes(model):
    for name, tensor in model.state_dict().items():
        print(name, tensor.size())

def convert_config(src_config: PretrainedConfig, translation_map: Dict[str, str], **kwargs) -> MemorizingTransformerConfig:
    """_summary_

    Args:
        translation_map (Dict[str, str]): _description_

    Returns:
        MemorizingTransformerConfig: _description_
    """
    src_config = src_config.to_dict()
    trgt_config = {}

    for src_key, val in src_config.items():
        if (trgt_key := translation_map.get(src_key)) is not None:
            trgt_config[trgt_key] = val
    
    trgt_config |= kwargs
    trgt_config = MemorizingTransformerConfig(**trgt_config)
    return trgt_config


def convert_weights(
        src_model: nn.Module,
        conversion_map: Dict[Union[str, re.Pattern], str],
        additional_funcs: List[Callable] = None
        ) -> Dict[str, Tensor]:
    
    src_state_dict = src_model.state_dict()
    trgt_state_dict= {}
    conversion_map = ConversionMap(conversion_map)
    counter = 0
    for src_name, tensor in src_state_dict.items():
        try:
            trgt_name = conversion_map[src_name]
            trgt_state_dict[trgt_name] = tensor
            counter += 1
        except KeyError:
            continue
    
    for func in additional_funcs:
        trgt_state_dict = func(src_state_dict, trgt_state_dict, counter)
    
    print(f"Converted {counter} entries. (N. Entries of SRCModel = {len(src_state_dict)})")
    return trgt_state_dict




def bert_join_key_value_projections(src_state_dict, trgt_state_dict, counter):
    """
    MemorizingTransformers use a joint layer to project keys and values.
    This functions fuses the independent layers in src model to a single layer.
    """
    layer_idc = sorted(list(set([
        name.split(".")[2]
        for name in src_state_dict.keys() if name.startswith("encoder.layer.")
        ])))
    for layer_idx in layer_idc:
        src_k_weights = src_state_dict[f"encoder.layer.{layer_idx}.attention.self.key.weight"]
        src_k_bias = src_state_dict[f"encoder.layer.{layer_idx}.attention.self.key.bias"]

        src_v_weights = src_state_dict[f"encoder.layer.{layer_idx}.attention.self.value.weight"]
        src_v_bias = src_state_dict[f"encoder.layer.{layer_idx}.attention.self.value.bias"]
        
        weights = torch.cat((src_k_weights, src_v_weights), dim=1)
        bias = torch.cat((src_k_bias, src_v_bias), dim=0)
        
        trgt_weight_name = f"encoder.layers.{layer_idx}.0.fn.to_kv.weight"
        trgt_state_dict[trgt_weight_name] = weights
        
        trgt_bias_name = f"encoder.layers.{layer_idx}.0.fn.to_kv.bias"
        trgt_state_dict[trgt_bias_name] = bias
        counter += 4

    return trgt_state_dict

# HParams of Bert that can be directly translated to MemeTRF-HParams.
BERT_CONFIG_TRANSLATE_MAP = {
    "hidden_size": "dim",
    "vocab_size": "num_tokens",
    "num_hidden_layers": "depth",
    "num_attention_heads": "heads",
    "hidden_dropout_prob": "ff_dropout",
    "intermediate_size": "intermediate_dim"
}
# Rules to convert Bert weights to corresponding memory model weights.
BERT_WEIGHT_CONVERSION_MAP = {
    # Token embeddings
    "embeddings.word_embeddings.weight": "encoder.token_emb.weight",
    # AttnLayer query projection
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.self.query.weight"): "encoder.layers.{layer_idx}.0.fn.to_q.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.self.query.bias"): "encoder.layers.{layer_idx}.0.fn.to_q.bias",
    # AttnLayer attention output projection + layer norm
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.dense.weight"): "encoder.layers.{layer_idx}.0.fn.to_out.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.dense.bias"): "encoder.layers.{layer_idx}.0.fn.to_out.bias",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.LayerNorm.weight"): "encoder.layers.{layer_idx}.1.norm.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).attention.output.LayerNorm.bias"): "encoder.layers.{layer_idx}.1.norm.bias",
    # AttnLayer intermediate projections
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).intermediate.dense.weight"): "encoder.layers.{layer_idx}.1.fn.net.0.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).intermediate.dense.bias"): "encoder.layers.{layer_idx}.1.fn.net.0.bias",
    # AttnLayer global output projections + layer norm
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.dense.weight"): "encoder.layers.{layer_idx}.1.fn.net.3.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.dense.bias"): "encoder.layers.{layer_idx}.1.fn.net.3.bias",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.LayerNorm.weight"): "encoder.layers.{layer_idx}.1.norm.weight",
    re.compile(r"encoder.layer.(?P<layer_idx>\d+).output.LayerNorm.bias"): "encoder.layers.{layer_idx}.1.norm.bias"
}

if __name__ == "__main__":
    
   bert_model = AutoModel.from_pretrained("deepset/gbert-large") 
   bert_tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
   print_sizes(bert_model)

   config = convert_config(
       bert_model.config,
       BERT_CONFIG_TRANSLATE_MAP,
       pad_id=bert_tokenizer.pad_token_id
    )
   model = MemorizingTransformerModel(config)
   #print(model.config)
   print_sizes(model)

   converted_state_dict = convert_weights(bert_model, BERT_WEIGHT_CONVERSION_MAP, additional_funcs=[bert_join_key_value_projections])
   model.load_state_dict(converted_state_dict, strict=False)

   model.save_pretrained("_test/mem-gbert-large")
   bert_tokenizer.model_max_length = None
   bert_tokenizer.init_kwargs["model_max_length"] = None
   bert_tokenizer.save_pretrained("_test/mem-gbert-large")


