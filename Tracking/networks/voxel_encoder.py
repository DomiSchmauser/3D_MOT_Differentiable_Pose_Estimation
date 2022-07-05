import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelEncoder(nn.Module):

    ''' 3D-convolutional encoder network for voxel input.
    Args:
        dim (int): input dimension 32 x 32 x 32
        c_dim (int): output dimension 9
    '''

    def __init__(self, input_channel=1, output_channel=9):
        super().__init__()
        self.relu = F.relu
        self.leaky_relu = F.leaky_relu

        self.conv_in = nn.Conv3d(input_channel, 8, 3, padding=1) # 1 x 32 x 32 x 32
        self.conv_0 = nn.Conv3d(8, 16, 3, padding=1, stride=2)
        self.conv_1 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv_2 = nn.Conv3d(32, 32, 3, padding=1, stride=2)
        self.fc = nn.Linear(32 * 4 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, output_channel)

    def forward(self, x):
        batch_size = x.size(0) # x_shape = BS x in_channels x 32 x 32 x 32

        net = self.conv_in(x)
        #print('l1',net.shape)
        net = self.conv_0(self.relu(net))
        #print('l2', net.shape)
        net = self.conv_1(self.relu(net))
        #print('l3', net.shape)
        net = self.conv_2(self.relu(net))
        #print('l4', net.shape)

        hidden = net.view(batch_size, 32 * 4 * 4 * 4)
        output = self.fc(self.leaky_relu(hidden))
        output = self.fc2(self.leaky_relu(output)) # BS x out_dim


        return output