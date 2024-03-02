import torchvision.transforms as transforms
from torchvision import datasets

transform = transforms.Compose([
    # transforms.HorizontalFlip(probability_goes_here),
    transforms.RandomHorizontalFlip(probability_goes_here)
    transforms.RandomGrayscale(probability_goes_here),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('data', train=True,
                              download=True,transform=transform)
test_data = datasets.CIFAR10('data', train=False,
                             download=True, transform=transform)


transform = {"train": transforms.Compose([transforms.RandomHorizontalFlip(probability_goes_here),
                        transforms.RandomGrayscale(probability_goes_here),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]),
             "test": transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         transforms.Resize(size_goes_here)])
            }

train_data = datasets.CIFAR10('data', train=True, download=True,
                              transform=transform["train"])

test_data = datasets.CIFAR10('data', train=True, download=True,
                             transform=transform['test'])
