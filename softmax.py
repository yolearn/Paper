import torch
import torch.nn as nn

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(out)
out.backward()

print(out.grad_fn)
print(x.grad)
