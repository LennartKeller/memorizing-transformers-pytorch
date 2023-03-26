import torch
from transformers import AutoTokenizer
from memorizing_transformers_pytorch import MemorizingTransformerConfig, MemorizingTransformerForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("_test/mem-bert-base-german-cased")
config = MemorizingTransformerConfig.from_pretrained("_test/mem-bert-base-german-cased")
model = MemorizingTransformerForMaskedLM.from_pretrained("_test/mem-bert-base-german-cased", config=config)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

inputs = tokenizer("Berlin ist die Hauptstadt von [MASK].", return_tensors="pt")
inputs["labels"] = inputs["input_ids"]
outputs = model(**inputs.to(model.device))
print(outputs)

logits = outputs["logits"]
print("Prediction for [MASK]-token:", tokenizer.decode([logits[0, -2].argmax()]))