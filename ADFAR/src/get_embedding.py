from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, BertConfig

model_type = 'dnabert2'
mlm_path = 'zhihan1996/DNABERT-2-117M'
config_tgt = BertConfig.from_pretrained(mlm_path)
mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_path, config=config_tgt, trust_remote_code=True)
mlm_model.to('cuda')

tokenizer_mlm = AutoTokenizer.from_pretrained(
        mlm_path,
        model_max_length=128,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

embedding_layer = mlm_model.bert.embeddings.word_embeddings

# Get embeddings for all tokens in the vocabulary
vocab_size = tokenizer_mlm.vocab_size
all_embeddings = embedding_layer.weight.data  # Shape: [vocab_size, hidden_dim]

# If you want to get embeddings for specific tokens:
token_ids = list(range(vocab_size))  # All token IDs in the vocabulary

# To get embeddings for individual tokens (example):
for token_id in token_ids:
    token_embedding = embedding_layer.weight.data[token_id]
    token_str = tokenizer_mlm.convert_ids_to_tokens(token_id)
    # Do something with the embedding and token string

# If you need to convert embeddings to CPU/numpy format:
import torch
if torch.cuda.is_available():
    all_embeddings = all_embeddings.cpu()
embeddings_numpy = all_embeddings.numpy()

output_file = f"subword_{model_type}_embeddings.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for token_id in range(vocab_size):
        token_embedding = embeddings_numpy[token_id]
        token_str = tokenizer_mlm.convert_ids_to_tokens(token_id)
        embedding_str = " ".join(map(str, token_embedding))
        f.write(f"{token_str} {embedding_str}\n")

print(f"Embeddings saved to {output_file}")