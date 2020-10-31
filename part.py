import torch
import torch.nn as nn
import copy

def clone(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for i in range(n)])

    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, sublayer, ffn):
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.sublayer = clone(SublayerConnection(512, nn.dropout(0.1)), 3)
        self.ffn = ffn
        
    def forward(self, x):
        pass

a = torch.tensor([1,2,3])
print(a.shape)