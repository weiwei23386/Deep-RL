"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    return mlp
# Example usage
# input_size = 10
# output_size = 5
# n_layers = 3
# size = 20

# mlp = build_mlp(input_size, output_size, n_layers, size)
# print(mlp)


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    # a metaclass is a class of a class that defines how a class behaves. 
    # A class is an instance of a metaclass. 
    # By default, the metaclass for all classes in Python is `type`.
    # This will raise an error because MyAbstractClass cannot be instantiated
    # abstract_instance = MyAbstractClass() (wrong)
    # This is useful for ensuring that any specific policy implementations 
    # adhere to a consistent interface.

    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        # collects any additional keyword arguments that are not 
        # explicitly listed in the function's parameters into a dictionary.
        # This is useful when the class being defined inherits from 
        # a parent class that also takes additional arguments.

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        # the model outputs both the mean (mean_net) and 
        # the logarithm of the standard deviation (logstd) of a Gaussian distribution.
        self.mean_net.to(ptu.device)
        self.logstd = nn.Parameter(
            # there is one standard deviation parameter for each action dimension
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            # to concatenate the logstd parameter (wrapped in a list) with the parameters of mean_net.
            # Using a list ensures that logstd is treated as an iterable
            # the `output size` of self.mean_net is equal to self.logstd`(3,)`
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # Query mean_net to get mean values
        mean = self.mean_net(observation)

        # Create a Gaussian distribution with mean an std_deviation
        # For every action there is a Gaussian distribution
        std_deviation = torch.exp(self.logstd)
        distribution = distributions.Normal(mean, std_deviation)

        # Sample an action from the distribution
        action = distribution.sample()

        return action
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!
        # raise NotImplementedError

    def update(self, observations, actions): # 
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy
        :param actions: actions we want the policy to imitate
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss
        action_preds = self.forward(observations)
        loss = torch.nn.MSELoss()(action_preds, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
