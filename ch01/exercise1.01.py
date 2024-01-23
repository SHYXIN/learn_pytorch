import torch

tensor_1 = torch.tensor([0.1,1,0.9,0.7,0.3])
tensor_2 = torch.tensor([[0,0.2,0.4,0.6],[1,0.8,0.6,0.4]])
tensor_3 = torch.tensor([[[0.3,0.6],[1,0]], [[0.3,0.6],[0,1]]])

print(tensor_1.shape)
print(tensor_2.shape)
print(tensor_3.shape)
