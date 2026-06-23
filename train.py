import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            X=[]
            Y = []
            torch.manual_seed(epoch)

            for i in range(batch_size):
                idx = torch.randint(0, len(data)-context_length, (1,)).item()
                X.append(data[idx : idx + context_length])
                Y.append(data[idx + 1: idx + 1 + context_length])
            
            X = torch.stack(X)
            Y = torch.stack(Y)
            logits = model(X)
            B,T,C = logits.shape
            logits_flat = logits.reshape(B*T,C)
            targets_flat = Y.reshape(B * T)
            loss = F.cross_entropy(logits_flat,targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        return round(loss.item(),4)



