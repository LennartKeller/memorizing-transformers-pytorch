import torch
import torch.nn.functional as F
import platform
from transformers import BertTokenizerFast, pipeline
from memorizing_transformers_pytorch import BertForMaskedLM

def make_fill_mask(model, tokenizer, device):
    mask_token_id = tokenizer.mask_token_id
    def fill_mask(texts):
        if isinstance(texts, str):
            texts = (texts, )
        with model.knn_memories_context(batch_size=1) as knn_memories:
            sequences = []
            for text_idx, text in enumerate(texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                inputs = inputs.to(device)
                inputs["knn_memories"] = knn_memories
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs["logits"]
                probs = F.softmax(logits, dim=-1)

                input_ids = inputs["input_ids"]
                if (input_ids == mask_token_id).long().sum() > 1:
                    raise ValueError("Only a single mask token is allowed")
                mask_token_probs = probs[..., input_ids == mask_token_id, :].view(-1)
                topk = mask_token_probs.topk(k=10)
                top_probs, top_token_ids = topk
                
                for token_id, prob in zip(top_token_ids, top_probs):
                    filled_input_ids = input_ids.clone()
                    filled_input_ids[mask_token_id == filled_input_ids] = token_id
                    filled_text = tokenizer.decode(filled_input_ids.view(-1), skip_special_tokens=True)
                    sequences.append({
                        "idx": text_idx,
                        "pred": tokenizer.decode(token_id),
                        "text": filled_text,
                        "prob": prob,
                        "input": text
                    })
        return sequences
    return fill_mask



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "mps" if platform.machine() == "arm64" else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-german-cased")
    model = BertForMaskedLM.from_pretrained("bert-base-german-cased").to(device)

    batch_size = 8
    inputs = tokenizer(["Das ist ein Test"] * batch_size, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    with model.knn_memories_context(batch_size=batch_size) as knn_memories:
        inputs = inputs.to(device)
        inputs["knn_memories"] = knn_memories
        outputs = model(**inputs)
    print(outputs["loss"])

    fill_mask = make_fill_mask(model=model, tokenizer=tokenizer, device=device)
    mask_token = tokenizer.mask_token
    sents = (
        f"Ich reite auf meinem {mask_token}.",
        f"Berlin ist die Hauptstadt von {mask_token}.",
        f"Wolfgang Amadeus {mask_token} war ein berühmter Komponist.",
        (f"Der Hund bellt {mask_token}.", f"Der Hund {mask_token} laut."),
        (f"Meine Name {mask_token} Thomas Müller.", f"Ich heiße Thomas {mask_token}."),
        (f"Meine Name {mask_token} Peter Schmidt.", f"Ich heiße Peter {mask_token}."),
        (f"Meine Name {mask_token} Sarah Fisch.", f"Ich heiße Sarah {mask_token}."),
    )
    for sent in sents:
        print(f"Predictions for {repr(sent)}")
        print(*fill_mask(sent), sep="\n")
        print("\n\n")
