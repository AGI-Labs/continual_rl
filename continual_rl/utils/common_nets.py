import torch
import torch.nn as nn


def get_network_for_size(size):
    """
    Size is expected to be [channel, dim, dim]
    """
    if isinstance(size, dict):
        image_size = size["image"]
        state_size = size["state_vector"]
    else:
        image_size = size
        state_size = 0

    image_size = list(image_size)  # In case the input is a tuple
    if len(image_size) == 1:
        net = StateSpaceNet
    elif image_size[-2:] == [7, 7]:
        net = ConvNet7x7
    elif image_size[-2:] == [28, 28]:
        net = ConvNet28x28
    elif image_size[-2:] == [84, 84]:
        net = ConvNet84x84
    elif image_size[-2:] == [64, 64]:
        # just use 84x84, it should compute output dim
        net = ConvNet84x84
    else:
        raise AttributeError(f"Unexpected input size: {size}")

    return net(image_size, state_size)


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
        if isinstance(x, dict):
            x_image = x["image"]
            x_state = x["state_vector"]
        else:
            x_image = x
            x_state = None

        x = self._conv_net(x_image.float())

        if x_state is not None:
            x = torch.cat((x, x_state), dim=-1)

        x = self._post_flatten(x)
        return x


class ConvNet84x84(CommonConv):
    def __init__(self, image_observation_shape, state_shape):
        # This is the same as used in AtariNet in Impala (torchbeast implementation)
        output_size = 512
        conv_net = nn.Sequential(
                                  nn.Conv2d(in_channels=image_observation_shape[0], out_channels=32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                                  nn.ReLU(),
                                  nn.Flatten())
        intermediate_dim = ModelUtils.compute_output_size(conv_net, image_observation_shape) + state_shape[0]
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet28x28(CommonConv):
    def __init__(self, image_observation_shape, state_shape):
        output_size = 32
        conv_net = nn.Sequential(
            nn.Conv2d(image_observation_shape[0], 24, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),  # TODO: this is new... (check)
            nn.Conv2d(24, 48, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, image_observation_shape) + state_shape[0]
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class ConvNet7x7(CommonConv):
    def __init__(self, image_observation_shape, state_shape):
        # From: https://github.com/lcswillems/rl-starter-files/blob/master/model.py, modified by increasing each
        # latent size (2x)
        output_size = 64
        conv_net = nn.Sequential(
            nn.Conv2d(image_observation_shape[0], 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )
        intermediate_dim = ModelUtils.compute_output_size(conv_net, image_observation_shape) + state_shape[0]
        post_flatten = nn.Linear(intermediate_dim, output_size)
        super().__init__(conv_net, post_flatten, output_size)


class StateSpaceNet(nn.Module):
    def __init__(self, observation_shape, output_shape=None, **kwargs):
        super().__init__()
        self.output_shape = (64,) if output_shape is None else output_shape
        self.output_size = self.output_shape[0]

        num_state_space_layers = kwargs.pop("num_state_space_intermediate_layers", 1)
        hidden_dim = kwargs.pop("hidden_dim", 64)

        # Fake a first layer ... TODO: still desirable?
        hidden_dims = [hidden_dim for _ in range(num_state_space_layers+1)]
        layers = [nn.Linear(observation_shape[0], hidden_dims[0]), nn.ReLU(),]

        for layer_id in range(num_state_space_layers):
            layers.append(nn.Linear(hidden_dims[layer_id], hidden_dims[layer_id+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, self.output_shape[0]))

        self._net = nn.Sequential(*layers)

    def forward(self, x):
        return self._net(x.float())
