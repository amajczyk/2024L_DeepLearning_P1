# Define the CNN architecture - from ChatGPT
# ok, i would like a function, which will return a CNN object, given the consecutive kernel sizes, assuming that the pictures have 3 channels, and in every convolution layer we have the same number of ffilters. also after a conv layer there always is a max pooling layer (with the same stride and size) and a relu layer; simmilar for the fully connected layers
from typing import Union
import numpy as np
import torch
import torch.nn as nn



def create_cnn(
    kernel_sizes: Union[list, np.array] = [6, 6],
    num_filters: int = 16,
    filter_stride: int = 1,
    filter_padding: int = 1,

    pooling_size: int = 2,
    pooling_stride: int = 1,

    num_fc_layers: int = 2,
    fc_size: int = 10,

    image_size: int = 32,
    num_input_channels: int = 3,
    num_classes: int = 10,

):
    """
    Create a CNN with the given architecture.

    Args:
    kernel_sizes (list): A list of kernel sizes for each convolutional layer.
    num_filters (int): The number of filters in each convolutional layer.
    filter_stride (int): The stride for the convolutional layers.
    filter_padding (int): The padding for the convolutional layers.

    pooling_size (int): The size of the pooling layers.
    pooling_stride (int): The stride for the pooling layers.

    num_fc_layers (int): The number of fully connected layers (hidden).
    fc_size (int): The size of the fully connected layers.
    
    image_size (int): The size of the input images (assumed to be square).
    num_input_channels (int): The number of input channels.
    num_classes (int): The number of output classes.

    Returns:
    CNN: A PyTorch CNN model.
    """

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.num_conv_layers = len(kernel_sizes)
            self.num_fc_layers = num_fc_layers

            # Convolutional layers
            self.conv_layers = nn.ModuleList()
            self.pooling_layers = nn.ModuleList()
            self.relu_layers = nn.ModuleList()
            in_channels = num_input_channels
            for kernel_size in kernel_sizes:
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels,
                        num_filters,
                        kernel_size=kernel_size,
                        padding=filter_padding,
                        stride=filter_stride,
                    )
                )
                self.pooling_layers.append(
                    nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_stride)
                )
                self.relu_layers.append(nn.ReLU())
                in_channels = num_filters

            # Fully connected layers
            self.fc_layers = nn.ModuleList()
            self.fc_relu_layers = nn.ModuleList()
            

            spatial_dim = image_size

            for i, _ in enumerate(range(self.num_conv_layers)):
                spatial_dim = (spatial_dim - kernel_sizes[i] + 2 * filter_padding) // filter_stride + 1
                spatial_dim = (spatial_dim - pooling_size) // pooling_stride + 1

            # Calculate the input size for the fully connected layers
            input_size = spatial_dim * spatial_dim * num_filters


            for _ in range(num_fc_layers):
                self.fc_layers.append(nn.Linear(input_size, fc_size))
                self.fc_relu_layers.append(nn.ReLU())
                input_size = fc_size

            # Final output layer
            self.output_layer = nn.Linear(fc_size, num_classes)

        def forward(self, x):
            # Convolutional layers
            # i = 0
            for conv_layer, pooling_layer, relu_layer in zip(
                self.conv_layers, self.pooling_layers, self.relu_layers
            ):
                x = conv_layer(x)
                x = pooling_layer(x)
                x = relu_layer(x)
                # print(f'After conv_layer {i+1}: {x.size()}')
                # i += 1

            # Flatten the output for fully connected layers
            x = torch.flatten(x, 1)

            # Fully connected layers
            # i= 0
            for fc_layer, fc_relu_layer in zip(self.fc_layers, self.fc_relu_layers):
                x = fc_layer(x)
                x = fc_relu_layer(x)
                # print(f'After fc_layer {i+1}: {x.size()}')
                # i += 1

            # Output layer
            x = self.output_layer(x)
            # print(f'Output size: {x.size()}')
            return x

    return CNN()