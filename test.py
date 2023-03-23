import torch
from transformers import AutoTokenizer
from memorizing_transformers_pytorch import MemorizingTransformerConfig, MemorizingTransformerForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("_test/mem-gbert-large")
config = MemorizingTransformerConfig.from_pretrained("_test/mem-gbert-large")
model = MemorizingTransformerForMaskedLM.from_pretrained("_test/mem-gbert-large", config=config)
inputs = tokenizer("Paris ist die Hauptstadt von Frankreich.", return_tensors="pt")
inputs["labels"] = inputs["input_ids"]

print(model(**inputs.to(model.device)))