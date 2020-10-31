import torch
import torch.nn as nn

embedding = nn.Embedding(10, 3)
input = torch.zeros(2, 4, dtype=torch.long)
output = embedding(input)
print(input.size())
print(output.size())