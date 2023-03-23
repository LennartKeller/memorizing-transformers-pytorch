import re
import torch
from memorizing_transformers_pytorch import MemorizingTransformer
from transformers import AutoModel


def print_sizes(state_dict):
    for name, tensor in state_dict.items():
        print(name, tensor.size())


GPT2MODEL_CONVERSION_MAP = {
    # Token embeddings
    "wte.weight": "token_emb.weight",
    # AttnLayer Query Projection

}

if __name__ == "__main__":
    
    model = MemorizingTransformer(
        num_tokens = 50265,                 # number of tokens
        dim = 768,                          # dimension
        dim_head = 768 // 12,               # dimension per attention head
        heads=12,
        depth = 11,                          # number of layers
        memorizing_layers = (6, 9),         # which layers to have ANN memories
        max_knn_memories = 64000,           # maximum ANN memories to keep (once it hits this capacity, it will be reset for now, due to limitations in faiss' ability to remove entries)
        num_retrieved_memories = 32,        # number of ANN memories to retrieve
        clear_memories_on_sos_token_id = 1, # clear passed in ANN memories automatically for batch indices which contain this specified SOS token id - otherwise, you can also manually iterate through the ANN memories and clear the indices before the next iteration
    )

    gpt_model = AutoModel.from_pretrained("dbmdz/german-gpt2")

    # data = torch.randint(0, 20000, (2, 1024)) # mock data

    # knn_memories = model.create_knn_memories(batch_size = 2) # create collection of KNN memories with the correct batch size (2 in example)

    # logits = model(data, knn_memories = knn_memories) # (1, 1024, 20000)

    print_sizes(gpt_model.state_dict())
    print("_"*60)
    print_sizes(model.state_dict())