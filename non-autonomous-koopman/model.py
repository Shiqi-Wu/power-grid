import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
torch.set_default_dtype(torch.float64)


import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_dim, dictionary_dim=128):
        super(ResNet, self).__init__()
        self.in_features = input_dim
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0])
        self.layer2 = self._make_layer(block, 32, num_blocks[1])
        self.layer3 = self._make_layer(block, 64, num_blocks[2])
        self.linear = nn.Linear(64, dictionary_dim)

    def _make_layer(self, block, out_features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, out_features))
            self.in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.linear(out)
        return out

class TrainableDictionary(nn.Module):
    """
    A neural network module that builds a trainable dictionary for input data.
    Args:
        inputs_dim (int): The dimension of the input data.
        dictionary_dim (int): The dimension of the dictionary output.
        resnet_params (tuple): Parameters to define the ResNet structure.
    Methods:
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x` and returns the concatenated tensor
            of ones, the input `x`, and the dictionary output.
    """

    def __init__(self, inputs_dim, dictionary_dim, num_blocks):
        super(TrainableDictionary, self).__init__()
        self.inputs_dim = inputs_dim
        self.dictionary_dim = dictionary_dim
        
        # Initialize the ResNet model
        self.resnet = ResNet(
            block=BasicBlock,
            num_blocks=num_blocks,
            input_dim=inputs_dim,
            dictionary_dim=dictionary_dim
        )
        
    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device)
        # Pass input through ResNet
        dic = self.resnet(x)
        # Concatenate ones, input, and dictionary output
        y = torch.cat((ones, x), dim=1)
        y = torch.cat((y, dic), dim=1)
        return y

    
class BlockDiagonalKoopman(nn.Module):
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
        device = self.blocks[0].device
        K = torch.zeros(self.koopman_dim, self.koopman_dim, device=device)

        for i in range(self.num_blocks):
            angle = self.blocks[i]
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            K[2 * i, 2 * i] = cos
            K[2 * i, 2 * i + 1] = -sin
            K[2 * i + 1, 2 * i] = sin
            K[2 * i + 1, 2 * i + 1] = cos

        if self.koopman_dim % 2 != 0:
            K[-1, -1] = 1

        return K

    
    def forward_V(self):
        return self.V
    
class TimeEmbeddingBlockDiagonalKoopman(nn.Module):
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

    def __init__(self, dictionary_dim, inputs_dim, num_blocks):
        super(TimeEmbeddingBlockDiagonalKoopman, self).__init__()
        self.dictionary_dim = dictionary_dim
        self.koopman_dim = dictionary_dim + 1 + inputs_dim
        self.inputs_dim = inputs_dim
        self.num_blocks = num_blocks
        self.build()

    def build(self):
        # Initialize the TrainableDictionary with the updated signature
        self.dictionary = TrainableDictionary(
            inputs_dim=self.inputs_dim,
            dictionary_dim=self.dictionary_dim,
            num_blocks=self.num_blocks
        )
        # Initialize the Koopman operator
        self.koopman = BlockDiagonalKoopman(self.koopman_dim)
    
    def forward(self, x_dic):
        K = self.koopman.forward_K()
        V = self.koopman.forward_V()
        y = torch.matmul(x_dic, V)
        y = torch.matmul(y, K)
        # print(f"y.device: {y.device}, K.device: {K.device}")
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
    
    def regularization_loss(self):
        norm = torch.norm(self.koopman.V, p='fro')
        inv_nomr = torch.norm(torch.inverse(self.koopman.V), p='fro')
        condition_number = norm * inv_nomr
        return 0.00001 * condition_number
    

class KoopmanOperatorWithInputs(nn.Module):
    def __init__(self, koopman_dim, u_dim, num_blocks):
        super(KoopmanOperatorWithInputs, self).__init__()
        self.koopman_dim = koopman_dim
        self.u_dim = u_dim
        self.num_blocks = num_blocks
        self.build()

    def build(self):
        self.operator_nn = ResNet(
            block=BasicBlock,
            num_blocks=self.num_blocks,
            input_dim=self.u_dim,
            dictionary_dim=self.koopman_dim ** 2
        )

    def forward(self, u):
        k_vector = self.operator_nn(u)
        K = k_vector.view(-1, self.koopman_dim, self.koopman_dim)
        return K
    
class KoopmanModelWithInputs(nn.Module):
    def __init__(self, koopman_dim, u_dim, num_blocks, blockdiagonalkoopmanmodel):
        super(KoopmanModelWithInputs, self).__init__()
        self.koopman_dim = koopman_dim
        self.inputs_dim = u_dim
        self.num_blocks = num_blocks
        self.blockdiagonalkoopmanmodel = blockdiagonalkoopmanmodel
        self.build()
    
    def build(self):
        self.koopman = KoopmanOperatorWithInputs(
            koopman_dim=self.koopman_dim,
            u_dim=self.inputs_dim,
            num_blocks=self.num_blocks
        )
    
    def forward(self, x_dic, u):
        K = self.koopman(u)
        y_dic = torch.bmm(x_dic.unsqueeze(1), K).squeeze(1)
        return y_dic
    
def extract_x(x_dic, N, pca_dim):
    x = x_dic[:, (N-1) * pca_dim + 1: N * pca_dim + 1]
    return x

class state_input_network(nn.Module):
    def __init__(self, koopman_dim, u_dim, num_blocks):
        super(state_input_network, self).__init__()
        self.koopman_dim = koopman_dim
        self.u_dim = u_dim
        self.num_blocks = num_blocks
        self.build()

    def build(self):
        self.operator_nn = ResNet(
            block=BasicBlock,
            num_blocks=self.num_blocks,
            input_dim=self.u_dim + self.koopman_dim,
            dictionary_dim=self.koopman_dim
        )

    def forward(self, x_dic, u):
        xx = torch.cat((x_dic, u), dim=1)
        yy = self.operator_nn(xx)
        return yy
