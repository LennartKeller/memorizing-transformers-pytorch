import torch
from transformers import AutoTokenizer
from memorizing_transformers_pytorch import MemorizingTransformerConfig, MemorizingTransformerForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("_test/mlm-test/checkpoint-2000")
config = MemorizingTransformerConfig.from_pretrained("_test/mlm-test/checkpoint-2000")
model = MemorizingTransformerForMaskedLM.from_pretrained("_test/mlm-test/checkpoint-1500", config=config)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

inputs = tokenizer("Berlin ist die Hauptstadt von [MASK].", return_tensors="pt")
inputs["labels"] = inputs["input_ids"]
outputs = model(**inputs.to(model.device))
print(outputs)

logits = outputs["logits"]
print("Prediction for [MASK]-token:", tokenizer.decode([logits[0, -2].argmax()]))