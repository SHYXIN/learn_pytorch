# Exercise Loading a Pre-Trained Model in PyTorch

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
# content_img = image_loader("images/dog.jpg").to(device)
# style_img = image_loader("images/matisse.jpg").to(device)

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
# def tensor2image(tensor):
#     image = tensor.to('cpu').clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#
#     return image

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


# 1. Open the notebook from the previous exercise.

# 2. Load the VGG-19 pre-trained model from PyTorch:
model = models.vgg19(pretrained=True).features
# Select the features portion of the model, as explained previously. This will give
# you access to all the convolutional and pooling layers of the model, which are
# to be used to perform the extraction of features in subsequent exercises of
# this chapter.

# 3. Perform a for loop through the parameters of the previously loaded model. Set
# each parameter so that it doesn't require gradients calculations:
for param in model.parameters():
    param.requires_grad_(False)

# By setting the calculation of gradients to False, we ensure that no gradients are
# calculated during the training process.

# If your machine has a GPU available, add the following code snippet to the
# preceding snippet in order to allocate the model to the GPU:
# model.to(device)

# With that, you have successfully loaded a pre-trained model.

# 2. Print the architecture of the model we loaded in the previous exercise. This will
# help us identify the relevant layers so that we can perform the style transfer task:

print(model)

# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
#   (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (17): ReLU(inplace=True)
#   (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True)
#   (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (24): ReLU(inplace=True)
#   (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (26): ReLU(inplace=True)
#   (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)
#   (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (31): ReLU(inplace=True)
#   (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (33): ReLU(inplace=True)
#   (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (35): ReLU(inplace=True)
#   (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )

# 3. Create a dictionary for mapping the index of the relevant layers (keys) to a name
# (values). This will facilitate the process of calling relevant layers in the future:
relevant_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1',
                   '21': 'conv4_2', '28': 'conv5_1'}

# To create the dictionary, we use the output from the previous step, which
# displays each of the layers in the network. There, it is possible to observe that
# the first layer of the first stack is labeled 0, while the first layer of the second
# stack is labeled 5, and so on.

# 4. Create a function that will extract the relevant features (features extracted from
# the relevant layers only) from an input image. Name it features_extractor
# and make sure it takes the image, the model, and the dictionary we created
# previously as inputs:

def features_extractor(x, model, layers):

    features = {}
    for index, layer in model._modules.items():
        x = layer(x)
        if index in layers:
            features[layers[index]] = x

    return features
# The output should be a dictionary, with the keys being the name of the layer and
# the values being the output features from that layer.


# 5. Call the features_extractor function over the content and style images we
# loaded in the first exercise of this chapter:
content_features = features_extractor(content_img, model,
                                      relevant_layers)

style_features = features_extractor(style_img, model,
                                    relevant_layers)

# Perform the gram matrix calculation over the style features. Consider that the
# style features were obtained from different layers, which is why different gram
# matrices should be created, one for each layer's output:
style_grams = {}
for i in style_features:
    layer = style_features[i]
    _, d1, d2, d3 = layer.shape
    features = layer.view(d1, d2 * d3)
    gram = torch.mm(features, features.t())
    style_grams[i] = gram

# For each layer, the shape of the style features' matrix is obtained in order to
# vectorize it. Next, the gram matrix is created by multiplying the vectorized output
# by its transposed version.

# 7. Create an initial target image. This image will be compared against the content
# and style images later and be changed until the desired similarity is achieved:
target_img = content_img.clone().requires_grad_(True)

# It is good practice to create the initial target image as a copy of the content
# image. Moreover, it is essential to set it to require the calculation of gradients,
# considering that we want to be able to modify it in an iterative process until
# the content is similar to that of the content image and the style to that of the
# style image.

# Again, if your machine has a GPU available, make sure that you allocate the
# target image to the GPU as well, using the following code snippet instead:

# target_img = content_img.clone().requires_grad_(True).to(device)

# 8. Using the tensor2image function we created during the first exercise of this
# chapter, plot the target image, which should look the same as the content image:
plt.figure()
plt.imshow(tensor2image(target_img))
plt.title("Target Image")
plt.show()

# With that, you have successfully performed feature extraction and calculated the
# gram matrix to perform style transfer.
