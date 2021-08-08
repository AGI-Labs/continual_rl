import torch
import torch.nn as nn


def get_network_for_size(size):
    """
    Size is expected to be [channel, dim, dim]
    """
    size = list(size)  # In case the input is a tuple
    if size[-2:] == [7, 7]:
        net = ConvNet7x7
    elif size[-2:] == [28, 28]:
        net = ConvNet28x28
    elif size[-2:] == [84, 84]:
        net = ConvNet84x84
    elif size[-2:] == [64, 64]:
        # just use 84x84, it should compute output dim
        net = ConvNet84x84
    else:
        raise AttributeError("Unexpected input size")

    return net(size)


class ModelUtils(object):
    """
    Allows for images larger than their stated minimums, and will auto-compute the output size accordingly
    """
    @classmethod
    def compute_output_size(cls, net, observation_size):
        dummy_input = torch.zeros(observation_size).unsqueeze(0)  # Observation size doesn't include batch, so add it
        dummy_output = net(dummy_input).squeeze(0)  # Remove batch
        output_size = dummy_output.shape[0]
        return output_size


class CommonConv(nn.Module):
    def __init__(self, conv_net, post_flatten, output_size):
        super().__init__()
        self._conv_net = conv_net
        self._post_flatten = post_flatten
        self.output_size = output_size

    def forward(self, x):
        x = self._conv_net(x.float())
        x = self._post_flatten(x)
        return x


class ConvNet84x84(CommonConv):
    def __init__(self, observation_shape):
        # This is the same as used in AtariNet in Impala (torchbeast implementation)
        output_size = 512
        conv_net = nn.Sequential(
                                  nn.Conv2d(in_channels=observation_shape[0], out_channels=32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                  nn.ReLU(),
                                  nn.Flatten())
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet28x28(CommonConv):
    def __init__(self, observation_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 24, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),  # TODO: this is new... (check)
            nn.Conv2d(24, 48, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7(CommonConv):
    def __init__(self, observation_shape):
        # From: https://github.com/lcswillems/rl-starter-files/blob/master/model.py, modified by increasing each
        # latent size (2x)
        output_size = 64
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, observation_shape)
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)
