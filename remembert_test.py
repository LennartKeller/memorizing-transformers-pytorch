import torch
import torch.nn.functional as F
import platform
from transformers import BertTokenizerFast, pipeline
from memorizing_transformers_pytorch import RememBertForMaskedLM

def make_fill_mask(model, tokenizer):
    mask_token_id = tokenizer.mask_token_id
    
    def fill_mask(texts):

        model.eval()  
        if isinstance(texts, str):
            texts = (texts, )
        
        with model.knn_memories_context(batch_size=1) as knn_memories:
            sequences = []
            for text_idx, text in enumerate(texts):
                # Do not add CLS tokens on subsequent sentences to prevent memory clearing.
                if model.config.clear_memory_on_cls_token:
                    add_special_tokens = not text_idx
                else:
                    add_special_tokens = True 
                inputs = tokenizer(text, add_special_tokens=add_special_tokens,return_tensors="pt", truncation=True)
                inputs = inputs.to(model.device)
                inputs["knn_memories"] = knn_memories
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                logits = outputs["logits"]
                probs = F.softmax(logits, dim=-1)

                input_ids = inputs["input_ids"]
                if (input_ids == mask_token_id).long().sum() > 1:
                    raise ValueError("Only a single mask token is allowed")
                
                mask_token_probs = probs[..., input_ids == mask_token_id, :].view(-1)
                top_probs, top_token_ids = mask_token_probs.topk(k=10)
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
    
    tokenizer = BertTokenizerFast.from_pretrained("deepset/gbert-large")
    remembert_configs = {
        "memorizing_layers": [12, 22],
        "max_knn_memories": 32_000,
        "num_retrieved_memories": 32,
        "cls_token_id": tokenizer.cls_token_id,
        "knn_memory_multiprocessing": True,
        "normalize_memories": False,
        "clear_memory_on_cls_token": False
    }
    model = RememBertForMaskedLM.from_pretrained("deepset/gbert-large", **remembert_configs).to(device)

    batch_size = 8
    inputs = tokenizer(["Das ist ein Test"] * batch_size, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    with model.knn_memories_context(batch_size=batch_size) as knn_memories:
        inputs = inputs.to(device)
        inputs["knn_memories"] = knn_memories
        outputs = model(**inputs)
    print(outputs["loss"])

    fill_mask = make_fill_mask(model=model, tokenizer=tokenizer)
    mask_token = tokenizer.mask_token
    sents = (
        f"Ich reite auf meinem {mask_token}.",
        f"Berlin ist die Hauptstadt von {mask_token}.",
        
        f"Wolfgang Amadeus {mask_token} war ein berühmter Komponist.",
        f"Mozart starb in {mask_token}",
        (f"Wolfgang Amadeus {mask_token} war ein berühmter Komponist. Geboren wurde er in Salzburg.", f"Er starb in {mask_token}."),
        (f"Wolfgang Amadeus {mask_token} war ein berühmter Komponist.", f"Johann {mask_token} von Goethe war ein berühmter Schriftsteller."),
        
        (f"Der Hund {mask_token} laut.", f"Der Hund {mask_token} laut."),

        (f"Meine Name {mask_token} Thomas Müller.", f"Ich heiße Thomas {mask_token}."),
        (f"Meine Name {mask_token} Peter Schmidt.", f"Ich heiße Peter {mask_token}."),
        (f"Meine Name {mask_token} Sarah Fisch.", f"Ich heiße Sarah {mask_token}."),
        (f"Meine Name {mask_token} Sarah Hamid.", f"Ich heiße Sarah {mask_token}."),
    )
    for sent in sents:
        print(f"Predictions for {repr(sent)}")
        print(*fill_mask(sent), sep="\n")
        print("\n\n")

    print("Saving model to _test/test-saved-model")
    model.save_pretrained("_test/test-saved-model")
    tokenizer.save_pretrained("_test/test-saved-model")