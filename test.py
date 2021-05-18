import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# plt.show()

# x = torch.tensor([[1,3,4],[1,0,0]])
# print(x)
# print(x.unsqueeze(-2))

x = torch.rand((2,4,3,3))
m = torch.ByteTensor([[1,1,1],[1,0,0]])==0
print(x)
m = m.unsqueeze(-2).unsqueeze(1).repeat(1,4,3,1)
print(m)
print(m.shape)
print(x.masked_fill(m, 1e-9))
