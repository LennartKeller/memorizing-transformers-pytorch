import torch
from memorizing_transformers_pytorch import MemorizingTransformerEncoder, MemorizingTransformerConfig, MemorizingTransformerModel, MemorizingTransformerForMaskedLM

model = MemorizingTransformerEncoder(
    num_tokens = 20000,                 # number of tokens
    dim = 512,                          # dimension
    dim_head = 64,                      # dimension per attention head
    depth = 8,                          # number of layers
    memorizing_layers = (4, 5),         # which layers to have ANN memories
    max_knn_memories = 64000,           # maximum ANN memories to keep (once it hits this capacity, it will be reset for now, due to limitations in faiss' ability to remove entries)
    num_retrieved_memories = 32,        # number of ANN memories to retrieve
    clear_memories_on_sos_token_id = 1, # clear passed in ANN memories automatically for batch indices which contain this specified SOS token id - otherwise, you can also manually iterate through the ANN memories and clear the indices before the next iteration
)

data = torch.randint(0, 20000, (2, 1024)) # mock data

knn_memories = model.create_knn_memories(batch_size = 2) # create collection of KNN memories with the correct batch size (2 in example)

embeddings = model(data, knn_memories = knn_memories) # (2, 1024, 512)

config = MemorizingTransformerConfig(
    num_tokens = 20000,                 # number of tokens
    dim = 512,                          # dimension
    dim_head = 64,                      # dimension per attention head
    depth = 8,                          # number of layers
    memorizing_layers = (4, 5),         # which layers to have ANN memories
    max_knn_memories = 64000,           # maximum ANN memories to keep (once it hits this capacity, it will be reset for now, due to limitations in faiss' ability to remove entries)
    num_retrieved_memories = 32,        # number of ANN memories to retrieve
    clear_memories_on_sos_token_id = 1, # clear passed in ANN memories automatically for batch indices which contain this specified SOS token id - otherwise, you can also manually iterate through the ANN memories and clear the indices before the next iteration
)

hf_model = MemorizingTransformerModel(config)
hf_model.save_pretrained("_test/model")
hf_config = MemorizingTransformerConfig.from_pretrained("_test/model")
hf_model = MemorizingTransformerForMaskedLM.from_pretrained("_test/model", config=hf_config)
data = torch.randint(0, 20000, (2, 1024))
print("d", data.size())
outputs = hf_model(input_ids=data, labels=data)
print(outputs)