import torch
import torch.nn as nn




class USE_hyena(torch.nn.Module):
    def __init__(self, tokenizer, model):
        super(USE_hyena, self).__init__()
        self.tokenizer = tokenizer
        self.model = model.hyena  # Use the hyena part of the model
        print(self.model.device if hasattr(self.model, "device") else "cpu")

    def avg_pooling(self, model_output):
        # For HyenaDNA, we can use the last hidden state directly
        # The model_output is already the sequence of token embeddings
        # Simply take the mean of the sequence
        return torch.mean(model_output, dim=1)

    def forward(self, sents1, sents2):
        # Tokenize both lists of sentences
        inputs1 = self.tokenizer(
            sents1, padding=True, truncation=True, return_tensors="pt"
        )
        inputs2 = self.tokenizer(
            sents2, padding=True, truncation=True, return_tensors="pt"
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}

        with torch.no_grad():
            # Forward pass for both batches
            # For HyenaDNA, we need to get the embeddings first
           

            # Get the sequence embeddings
            outputs1 = self.model(inputs1["input_ids"])[0]  # Use the first output (hidden states)
            outputs2 = self.model(inputs2["input_ids"])[0]

        # Extract embeddings using average pooling
        embeddings1 = self.avg_pooling(outputs1)
        embeddings2 = self.avg_pooling(outputs2)

        # Compute cosine similarity between the sentence embeddings
        sim_scores = torch.nn.functional.cosine_similarity(
            embeddings1, embeddings2, dim=-1
        )
        return sim_scores

    def semantic_sim(self, sents1, sents2):
        # Return the similarity scores as a NumPy array
        return self.forward(sents1, sents2).cpu().numpy()

class USE_nt(torch.nn.Module):
    def __init__(self, tokenizer, model):
        super(USE, self).__init__()
        self.tokenizer = tokenizer
        self.model = model.esm
        print(self.model.device)

    def avg_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sents1, sents2):
        # Tokenize both lists of sentences in one go to optimize batching

        inputs1 = self.tokenizer(
            sents1, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        inputs2 = self.tokenizer(
            sents2, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            # Forward pass for both batches
            outputs1 = self.model(**inputs1)[0]
            outputs2 = self.model(**inputs2)[0]

        # Extract embeddings using average pooling
        embeddings1 = self.avg_pooling(outputs1, inputs1["attention_mask"])
        embeddings2 = self.avg_pooling(outputs2, inputs2["attention_mask"])

        # Compute cosine similarity between the sentence embeddings
        sim_scores = torch.nn.functional.cosine_similarity(
            embeddings1, embeddings2, dim=-1
        )
        return sim_scores

    def semantic_sim(self, sents1, sents2):
        # Return the similarity scores as a NumPy array
        return self.forward(sents1, sents2).cpu().numpy()

class USE_DNABERT(torch.nn.Module):
    def __init__(self, tokenizer, model):
        super(USE, self).__init__()
        self.tokenizer = tokenizer
        self.model = model.bert
        print(self.model.device)

    def avg_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sents1, sents2):
        # Tokenize both lists of sentences in one go to optimize batching
        
        inputs1 = self.tokenizer(sents1, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        inputs2 = self.tokenizer(sents2, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            # Forward pass for both batches
            outputs1 = self.model(**inputs1)[0]
            outputs2 = self.model(**inputs2)[0]
        
        # Extract embeddings using average pooling
        embeddings1 = self.avg_pooling(outputs1, inputs1['attention_mask'])
        embeddings2 = self.avg_pooling(outputs2, inputs2['attention_mask'])
        
        # Compute cosine similarity between the sentence embeddings
        sim_scores = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=-1)
        return sim_scores

    def semantic_sim(self, sents1, sents2):
        # Return the similarity scores as a NumPy array
        return self.forward(sents1, sents2).cpu().numpy()