import torch

tensor_1 = torch.tensor([0,1,1,0.9,0.7,0.3]).cuda()
tensor_2 = torch.tensor([[0,0.2,0.4,0.6],[1,0.8,0.6,0.4]]).cuda()
tensor_3 = torch.tensor([[[0.3,0.6],[1,0]], [[0.3,0.6],[0,1]]]).cuda()

print(tensor_1.shape)
print(tensor_2.shape)
print(tensor_3.shape)

x = torch.Tensor(10).random_(0, 10)
x.to("cuda")

example_1 = torch.randn(3,3)
example_2 = torch.randint(low=0, high=2, size=(3,3)).type(torch.FloatTensor)

print(example_1)
print(example_2)




