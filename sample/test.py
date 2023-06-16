import torch

a = torch.randint(1, 10, [2,10,3])
b = torch.randint(1, 10, [2, 3, 3])
c = torch.randint(1, 10, [2, 1, 3])
d = a @ b + c
print(d.shape)