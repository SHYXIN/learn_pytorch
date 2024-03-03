# Exercise 5.01: Loading and Displaying Images

# 1. Import all the packages that will be required to perform style transfer:
import numpy as np
import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If you have a GPU available, define a variable named device equal to cuda,
# which will be used to allocate some variables to the GPU of your machine:
device = "cuda" if torch.cuda.is_available() else 'cpu'
# device

# 2. Set the image size to be used for both images. Also, set the transformations
# to be performed over the images, which should include resizing the images,
# converting them into tensors, and normalizing them:
imsize = 224

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

print(torch.cuda.is_available())

# Using this code, the images are resized to the same size as the images that were
# originally used to train the VGG-19 model. Normalization is done using the same
# values that were used to normalize the training images as well.
# Note
# The VGG network was trained using normalized images, where each
# channel has a mean of 0.485, 0.456, and 0.406, respectively, and a
# standard deviation of 0.229, 0.224, and 0.225, respectively.


# 3. Define a function that will receive the image path as input and use PIL to open
# the image. Next, it should apply the transformations to the image:
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image

# 4. Call the function to load the content and style images. Use the dog image as
# the content image and the Matisse image as the style image, both of which are
# available in this book's GitHub repository:
content_img = image_loader("images/dog.jpg")
style_img = image_loader("images/matisse.jpg")

# If your machine has a GPU available, use the following code snippet instead to
# achieve the same results:
content_img = image_loader("images/dog.jpg").to(device)
style_img = image_loader("images/matisse.jpg").to(device)

# The preceding code snippet allocates the variables holding the images to the
# GPU so that all the operations using these variables are handled by the GPU.

# 5. To display the images, convert them back into PIL images and revert the
# normalization process. Define these transformations in a variable
unloader = transforms.Compose([
    transforms.Normalize((-0.485/0.229,-0.456/0.224, -0.406/0.225),
                         (1/0.229, 1/0.224, 1/0.225)),
    # 和上方为对应关系 mean是 -均值/std, 标准差为1/std
    #     transforms.Normalize((0.485, 0.456, 0.406),
    #                          (0.229, 0.224, 0.225))
    transforms.ToPILImage()
])

# 6. Create a function that clones the tensor, squeezes it, and applies the
# transformations defined in the previous step to the tensor:
def tensor2image(tensor):
    image = tensor.clone()
    image = image.squeeze(0)
    image = unloader(image)

    return image

# If your machine has a GPU available, use the following equivalent code
# snippet instead:
def tensor2image(tensor):
    image = tensor.to('cpu').clone()
    image = image.squeeze(0)
    image = unloader(image)

    return image

# The preceding code snippet allocates the images back to the CPU so that we can
# plot them.
# 7. Call the function for both images and plot the results
plt.figure()
plt.imshow(tensor2image(content_img))
plt.title("Content Image")
plt.show()

plt.figure()
plt.imshow(tensor2image(style_img))
plt.title("Style Image")
plt.show()
