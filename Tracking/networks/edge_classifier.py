import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Binary Classifier for classifying active/ non-active edges 
'''

class EdgeClassifier(nn.Module):

    def __init__(self, input_dim=32, intermed_dim=None):
        super(EdgeClassifier, self).__init__()
        if intermed_dim is None:
            self.fc1 = nn.Linear(input_dim, 16)
            self.fc2 = nn.Linear(16, 1)
        else:
            self.fc1 = nn.Linear(input_dim, intermed_dim)
            self.fc2 = nn.Linear(intermed_dim, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x