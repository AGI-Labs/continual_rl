import torch
import torch.nn as nn
from continual_rl.utils.utils import Utils


def get_network_for_size(size, output_shape=None, **kwargs):
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

    return net(size, output_shape, **kwargs)


class ModelUtils(object):
    """
    Allows for images larger than their stated minimums, and will auto-compute the output size accordingly
    """
    @classmethod
    def compute_output_shape(cls, net, observation_size):
        dummy_input = torch.zeros(observation_size).unsqueeze(0)  # Observation size doesn't include batch, so add it
        dummy_output = net(dummy_input).squeeze(0)  # Remove batch
        output_shape = dummy_output.shape
        return output_shape


class CommonConv(nn.Module):
    def __init__(self, conv_net, post_flatten, output_shape):
        super().__init__()
        self._conv_net = conv_net
        self._post_flatten = post_flatten
        self.output_shape = output_shape
        self.output_size = output_shape[0]  # TODO: legacy so I don't break everything, but we should switch to shape

        print(f"Created conv network with total parameters: {Utils.count_trainable_parameters(self)}")

    def forward(self, x):
        x = self._conv_net(x.float())
        x = self._post_flatten(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self._res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding="same")
        )

    def forward(self, x):
        out = self._res_block(x)
        return x + out


class ResidualBlock1d(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self._res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1,
                      padding="same")
        )

    def forward(self, x):
        out = self._res_block(x.unsqueeze(-1)).squeeze(-1)  # Add and remove a "length" dim (batch, channel, length) for 1d.
        return x + out


class ConvNet84x84(CommonConv):
    def __init__(self, observation_shape, output_shape=None, **kwargs):
        # This is the same as used in AtariNet in Impala (torchbeast implementation)
        hidden_dim = kwargs.pop("hidden_dim", 32)
        nonlinearity = kwargs.pop("nonlinearity", nn.ReLU(inplace=True))
        arch = kwargs.pop("arch", "orig")
        output_shape = (512,) if output_shape is None else output_shape

        if arch == "orig":
            conv_net = nn.Sequential(
                                      nn.Conv2d(in_channels=observation_shape[0], out_channels=hidden_dim, kernel_size=8, stride=4),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=4, stride=2),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1),
                                      nonlinearity,
                                      nn.Flatten())
            intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
            post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        elif arch == "8xorig":  # For procgen - ratio is a bit different for Atari because of the output size
            conv_net = nn.Sequential(
                                      nn.Conv2d(in_channels=observation_shape[0], out_channels=hidden_dim*6, kernel_size=8, stride=4),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim*6, out_channels=hidden_dim*12, kernel_size=4, stride=2),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim*12, out_channels=hidden_dim*12, kernel_size=3, stride=1),
                                      nonlinearity,
                                      nn.Flatten())
            intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
            post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        elif arch == "32xorig":  # For procgen - ratio is a bit different for Atari because of the output size
            conv_net = nn.Sequential(
                                      nn.Conv2d(in_channels=observation_shape[0], out_channels=hidden_dim*14, kernel_size=8, stride=4),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim*14, out_channels=hidden_dim*28, kernel_size=4, stride=2),
                                      nonlinearity,
                                      nn.Conv2d(in_channels=hidden_dim*28, out_channels=hidden_dim*28, kernel_size=3, stride=1),
                                      nonlinearity,
                                      nn.Flatten())
            intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
            post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        elif arch == "impala_res_cnn":
            # Based on https://arxiv.org/pdf/1802.01561.pdf
            layers = []
            hidden_dims = [observation_shape[0], hidden_dim, hidden_dim*2, hidden_dim*2]
            for layer_id in range(3):
                last_dim = hidden_dims[layer_id]
                dim = hidden_dims[layer_id+1]
                layers.extend([nn.Conv2d(last_dim, dim, kernel_size=3, stride=1, padding="same"),
                               nn.MaxPool2d(kernel_size=3, stride=2),
                               ResidualBlock(dim, kernel_size=3),
                               ResidualBlock(dim, kernel_size=3), ])

            conv_net = nn.Sequential(*layers,
                                     nn.Flatten())
            intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
            post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        elif arch == "none":
            conv_net = nn.Identity()
            output_shape = observation_shape
            post_flatten = nn.Identity()
        else:
            raise Exception(f"Unknown architecture {arch}")

        super().__init__(conv_net, post_flatten, output_shape)


class ConvNet28x28(CommonConv):
    def __init__(self, observation_shape, output_shape, **kwargs):
        output_shape = (32,) if output_shape is None else output_shape
        conv_net = nn.Sequential(
            nn.Conv2d(observation_shape[0], 24, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
        post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        super().__init__(conv_net, post_flatten, output_shape)


class ConvNet7x7(CommonConv):
    def __init__(self, observation_shape, output_shape=None, **kwargs):
        # From: https://github.com/lcswillems/rl-starter-files/blob/master/model.py, modified by increasing each
        # latent size (2x)
        output_shape = (64,) if output_shape is None else output_shape
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
        intermediate_dim = ModelUtils.compute_output_shape(conv_net, observation_shape)[0]
        post_flatten = nn.Linear(intermediate_dim, output_shape[0])
        super().__init__(conv_net, post_flatten, output_shape)
