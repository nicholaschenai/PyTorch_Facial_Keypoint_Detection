## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Done: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 4)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_drop = nn.Dropout2d()
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.conv4_drop = nn.Dropout2d()      
        self.fc1 = nn.Linear(43264, 1000)
        #self.fc1 = nn.Linear(6400, 1000)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1000, 136)


        
    def forward(self, x):
        ## Done: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x= self.conv1_drop(F.max_pool2d(F.relu(self.conv1(x)),2))
        x= self.conv2_drop(F.max_pool2d(F.relu(self.conv2(x)),2))
        x= self.conv3_drop(F.max_pool2d(F.relu(self.conv3(x)),2))
        x= self.conv4_drop(F.max_pool2d(F.relu(self.conv4(x)),2))

        x = x.view(-1,  self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    # This function is lifted from http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    # used to calculate dimensions after flattening
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features