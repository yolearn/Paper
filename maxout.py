import torch
import torch.nn as nn

class MaxOut(nn.Module):
    ''' Maxout layer'''

    def __init__(self, inp_dim, out_dim, pool_size):
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        '''
        super().__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.pool_size = pool_size
        self.linear = nn.Linear(inp_dim, out_dim * pool_size)

    def forward(self, x):
        '''
        Args:
            x: input (bs, input_dim)
        Return:
            a: output after activation (bs, output_dim)
        '''

        inp = x
        out = self.linear(x)
        out = out.view(-1, self.out_dim, self.pool_size)
        a, _ = torch.max(out, dim=-1, keepdim=False)

        return a

INPUT_DIM = 5
OUTPU_DIM = 3
BATCH_SIZE = 32
k = 4

inp = torch.rand(BATCH_SIZE, INPUT_DIM)
maxout = MaxOut(inp_dim=INPUT_DIM, out_dim=OUTPU_DIM, pool_size=k)
output = maxout(inp)
print(output.size())