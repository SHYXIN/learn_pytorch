# Activity 5.01: Performing Style Transfer
# 1. Import the required libraries:
import numpy as np
import torch
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, models

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# If your machine has a GPU available, make sure to define a variable named
# device that will help to allocate some variables to the GPU, as follows:

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Specify the transformations to be performed over the input images. Be sure to
# resize them to the same size, convert them into tensors, and normalize them:
imsize = 224

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# 3. Define an image loader function. It should open the image and load it. Call the
# image loader function to load both input images
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image

content_img = image_loader("images/landscape.jpg")
style_img = image_loader("images/monet.jpg")
# If your machine has a GPU available, use the following code snippet instead:
# def image_loader(image_name):
#     image = Image.open(image_name)
#     image = loader(image).unsqueeze(0)
#     return image
# content_img = image_loader("images/landscape.jpg").to(device)
# style_img = image_loader("images/monet.jpg").to(device)

# 4. To be able to display the images, set the transformations to revert the
# normalization of the images and to convert the tensors into PIL images:
unloader = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.2224, -0.406/0.225),
                         (1/0.229, 1/0.224, 1/0.225)),
    transforms.ToPILImage()
])

# 5. Create a function (tensor2image) that's capable of performing the previous
# transformation over tensors. Call the function for both images and plot
# the results:
def tensor2image(tensor):
    image = tensor.clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

plt.figure()
plt.imshow(tensor2image(content_img))
plt.title("Content Image")
plt.show()

plt.figure()
plt.imshow(tensor2image(style_img))
plt.title("Style Image")
plt.show()

# If your machine has a GPU available, use the following code snippet instead:
# def tensor2image(tensor):
#     image = tensor.to("cpu").clone()
#     image = image.squeeze(0)
#     image = unloader(image)
#     return image

# plt.figure()
# plt.imshow(tensor2image(content_img))
# plt.title("Content Image")
# plt.show()
#
# plt.figure()
# plt.imshow(tensor2image(style_img))
# plt.title("Style Image")
# plt.show()

# 6. Load the VGG-19 model:
model = models.vgg19(pretrained=True).features
for param in model.parameters():
    param.requires_grad_(False)

# If your machine has a GPU available, make sure to allocate the variable
# containing your model to the GPU, as follows:
# model.to(device)

# 7. Create a dictionary for mapping the index of the relevant layers (keys) to a name
# (values). Then, create a function to extract the feature maps of the relevant
# layers. Use them to extract the features of both input images.
# The following function should extract the features of a given image for each of
# the relevant layers:
relevant_layers = {'0': 'conv1_1', '5': 'conv2_1',
                   '10': 'conv3_1', '19': 'conv4_1',
                   '21': 'conv4_2', '28': 'conv5_1'}

def features_extractor(x, model, layers):
    features = {}
    for index, layer in model._modules.items():
        x = layer(x)
        if index in layers:
            features[layers[index]] = x
    return features

# Next, the function should be called for both the content and style images:
content_features = features_extractor(content_img,
                                      model,
                                      relevant_layers)
style_features = features_extractor(style_img, model,
                                    relevant_layers)

# 8. Calculate the gram matrix for the style features. Also, create the initial
# target image.
# The following code snippet creates the gram matrix for each of the layers that
# was used to extract style features:
style_grams = {}
for i in style_features:
    layer = style_features[i]
    _, d1, d2, d3 = layer.shape
    features = layer.view(d1, d2 * d3)
    gram = torch.mm(features, features.t())
    style_grams[i] = gram

# Next, the initial target image is created as a clone of the content image:
target_img = content_img.clone().requires_grad_(True)

# If your machine has a GPU available, use the following code snippet instead:
# target_img = content_img.clone().requires_grad_(True).to(device)

# 9. Set the weights for different style layers, as well as the weights for the content
# and style losses:
style_weights = {'conv1_1': 1., 'conv2_1': 0.8,
                 'conv3_1': 0.6, 'conv4_1': 0.4,
                 'conv5_1': 0.2}

alpha = 1
beta = 1e5

# 10. Run the model for 500 iterations. Define the Adam optimization algorithm before
# starting to train the model, using 0.001 as the learning rate:
print_statement = 500
optimizer = torch.optim.Adam([target_img], lr=0.001)
iterations = 5000

for i in range(1, iterations+1):
    # Extract features for all relevant layers
    target_features = features_extractor(target_img, model, relevant_layers)
    # Calculate the content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    # Loop through all style layers
    style_losses = 0
    for layer in style_weights:
        # Create gram matrix for that layer
        target_feature = target_features[layer]
        _, d1, d2, d3 = target_feature.shape

        target_reshaped = target_feature.view(d1, d2 * d3)
        target_gram = torch.mm(target_reshaped, target_reshaped.t())

        style_gram = style_grams[layer]

        # Calculate style loss for that layer
        style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)

        # Calculate style loss for all layers
        style_losses += style_loss

    # Calculate the total loss
    total_loss = alpha * content_loss + beta * style_losses

    # Perform back propagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print the target image
    if i % print_statement == 0 or i == 1:
        print('Total loss: ', total_loss.item())
        plt.imshow(tensor2image(target_img))
        plt.show()

# 10. Run the model for 500 iterations. Define the Adam optimization algorithm before
# starting to train the model, using 0.001 as the learning rate:
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(tensor2image(content_img))
ax2.imshow(tensor2image(target_img))
ax3.imshow(tensor2image(style_img))
plt.show()



