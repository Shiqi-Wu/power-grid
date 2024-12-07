import torch
import torch.nn as nn
import numpy as np
import os
torch.set_default_dtype(torch.float64)

class TrainableDictionary(nn.modules):
    """
    A neural network module that builds a trainable dictionary for input data.
    Args:
        inputs_dim (int): The dimension of the input data.
        dictionary_dim (int): The dimension of the dictionary output.
        layers_params (list of int): A list containing the number of units in each hidden layer.
        activation (nn.Module, optional): The activation function to use between layers. Default is nn.Tanh().
    Methods:
        build():
            Constructs the neural network layers based on the provided parameters.
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x` and returns the concatenated tensor
            of ones, the input `x`, and the dictionary output.
    """

    def __init__(self, inputs_dim, dictionary_dim, layers_params, activation = nn.Tanh()):
        super(TrainableDictionary, self).__init__()
        self.inputs_dim = inputs_dim
        self.dictionary_dim = dictionary_dim
        self.layers_params = layers_params
        self.activation = activation
        self.build()
        
    def build(self):
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.inputs_dim, self.layers_params[0]))
        self.layers.append(self.activation)
        for i in range(1, len(self.layers_params)):
            self.layers.append(nn.Linear(self.layers_params[i-1], self.layers_params[i]))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(self.layers_params[-1], self.dictionary_dim))
    
    def forward(self, x):
        ones = torch.ones(dic.shape[0], 1)
        dic = x
        for layer in self.layers:
            dic = layer(dic)
        y = torch.cat((ones, x), dim = 1)
        y = torch.cat((y, dic), dim = 1)
        return y
    
class BlockDiagonalKoopman(nn.modules):
    """
    A PyTorch module representing a block diagonal Koopman operator.
    Attributes:
        koopman_dim (int): The dimension of the Koopman operator.
        num_blocks (int): The number of 2x2 blocks in the Koopman operator.
        blocks (nn.ParameterList): A list of parameters representing the angles for each 2x2 block.
        V (nn.Parameter): A parameter representing the matrix V.
    Methods:
        build():
            Initializes the block diagonal Koopman operator with random angles and matrix V.
        forward_K():
            Constructs the block diagonal Koopman operator matrix K using the angles.
        forward_V():
            Returns the matrix V.
    """

    def __init__(self, koopman_dim):
        super(BlockDiagonalKoopman, self).__init__()
        self.koopman_dim = koopman_dim
        self.num_blocks = self.koopman_dim // 2 
        self.build()

    def build(self):
        self.blocks = nn.ParameterList()
        for _ in range(self.num_blocks):
            angle = nn.Parameter(torch.randn(1))
            self.blocks.append(angle)
        if self.koopman_dim % 2 != 0:
            self.blocks.append(nn.Parameter(torch.randn(1)))
        
        self.V = nn.Parameter(torch.randn(self.koopman_dim, self.koopman_dim))

    def forward_K(self):
        K = torch.zeros(self.koopman_dim, self.koopman_dim)
        for i in range(self.num_blocks):
            angle = self.blocks[i]
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            K[2*i, 2*i] = cos
            K[2*i, 2*i+1] = -sin
            K[2*i+1, 2*i] = sin
            K[2*i+1, 2*i+1] = cos
        if self.koopman_dim % 2 != 0:
            K[-1, -1] = 1
        return K
    
    def forward_V(self):
        return self.V
    
class TimeEmbeddingBlockDiagonalKoopman(nn.modules):
    """
    A PyTorch module that implements a time embedding block with a block diagonal Koopman operator.
    Attributes:
        koopman_dim (int): The dimension of the Koopman operator.
        inputs (int): The number of input features.
        layers_params (list): A list of parameters for the layers in the dictionary.
        activation (nn.Module): The activation function to use in the dictionary.
    Methods:
        build():
            Builds the trainable dictionary and block diagonal Koopman operator.
        forward(x):
            Performs a forward pass through the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying the Koopman operator.
        dictionary_V(x):
            Computes the dictionary and applies the V matrix of the Koopman operator.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after applying the V matrix.
    """

    def __init__(self, koopman_dim, inputs, layers_params, activation = nn.Tanh()):
        super(TimeEmbeddingBlockDiagonalKoopman, self).__init__()
        self.koopman_dim = koopman_dim
        self.inputs = inputs
        self.layers_params = layers_params
        self.activation = activation
        self.build()

    def build(self):
        self.dictionary = TrainableDictionary(self.inputs, self.koopman_dim, self.layers_params, self.activation)
        self.koopman = BlockDiagonalKoopman(self.koopman_dim)
    
    def forward(self, x_dic):
        K = self.koopman.forward_K()
        V = self.koopman.forward_V()
        y = torch.matmul(x_dic, V)
        y = torch.matmul(y, K)
        return y
    
    def dictionary_forward(self, x):
        x_dic = self.dictionary(x)
        y = self.forward(x_dic)
        return y
    
    def dictionary_V(self, x):
        x_dic = self.dictionary(x)
        V = self.koopman.forward_V()
        y = torch.matmul(x_dic, V)
        return y