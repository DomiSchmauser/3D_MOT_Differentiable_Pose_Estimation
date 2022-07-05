from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False, use_leaky_relu=True):
        super(MLP, self).__init__()

        if use_leaky_relu:
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for idx, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(self.activation)

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        output = self.layers(input)
        return output