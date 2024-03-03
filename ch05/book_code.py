
image = Image.open(image_name)
image = transformation(image).unsqueeze(0)


image = tensor.clone()
image = image.squeeze(0)

image_display = transforms.Compose([transforms.Normal(
        (-0.5/0.25, -0.5/0.25,-0.5/0.25, -0.5/0.25),
        (1/0.25, 1/0.25, 1/0.25),
    ),
    transforms.ToPILImage()])



from torchvision import models
from tensorflow.keras.applications import VGG19

model = models.vgg19(pretrained=True).features

for param in model.parameters():
    param.requires_grad_(False)




layers = {'0': 'conv1_1', '5': 'conv2_2', '10': 'conv3_1',
          '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}

features = {}

x = image

for index, layer in model._modules.items():
    x = layer(image)
    if index in layers:
        features[layers[index]] = x
