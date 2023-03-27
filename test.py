import sys
import torch
from transformers import AutoTokenizer
from memorizing_transformers_pytorch import MemorizingTransformerConfig, MemorizingTransformerForMaskedLM

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "_test/mem-bert-base-german-cased"
print(f"Using model {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = MemorizingTransformerConfig.from_pretrained(MODEL_PATH)
model = MemorizingTransformerForMaskedLM.from_pretrained(MODEL_PATH, config=config)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

inputs = tokenizer("Berlin ist die Hauptstadt von [MASK].", return_tensors="pt")
inputs["labels"] = inputs["input_ids"]
outputs = model(**inputs.to(model.device))
print(outputs)

logits = outputs["logits"]
print("Prediction for [MASK]-token:", tokenizer.decode([logits[0, -2].argmax()]))