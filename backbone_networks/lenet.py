from torch import nn


def LeNet(input_size=None, output=None):
    """

    :param input_size: input channels
    :param output: output channels
    :return: Sequential model
    """
    if input_size is None:
        input_size = input_size
    # if output is None:
    #     output = 10
    lenet = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=6, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, bias=False),
        nn.Tanh(),
        # nn.Flatten(),
        # nn.Linear(in_features=120, out_features=84, bias=False),
        # nn.Tanh(),
        # nn.Linear(in_features=84, out_features=output, bias=False),
    )

    return lenet


def LeNet_300_100(input_size=None, output=None):
    """

    :param input_size: Flat Input dimension
    :param output: Flat output dimension
    :return: Sequential model
    """
    if input_size is None:
        input_size = input_size
    # if output is None:
    #     output = 10

    lenet = nn.Sequential(
        nn.Linear(input_size, 300),
        nn.ReLU(),
        nn.Linear(300, 100),
        nn.ReLU(),
        # nn.Linear(100, output)
    )

    return lenet
