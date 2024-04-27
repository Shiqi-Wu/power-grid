import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

class MinMaxScalerLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(MinMaxScalerLayer, self).__init__()      
        self.min_val = nn.Parameter(torch.tensor(min_val, dtype=torch.float32), requires_grad=False)
        self.max_val = nn.Parameter(torch.tensor(max_val, dtype=torch.float32), requires_grad=False)

    def transform(self, x):
        denominator = self.max_val - self.min_val
        # Handle the case where min_val and max_val are equal
        denominator = torch.where(denominator == 0, torch.tensor(1e-8), denominator)
        return (x - self.min_val) / denominator
    
    def inverse_transform(self, input):
        return input * (self.max_val - self.min_val) + self.min_val

class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pca_matrix):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.transform.weight = nn.Parameter(pca_matrix, requires_grad=False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        self.inverse_transform.weight = nn.Parameter(pca_matrix.T, requires_grad=False)

    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0):
        super(ResNetBlock, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        out = F.tanh(self.layer1(x))
        out = self.dropout(out)
        out = self.layer2(out)
        out += identity  # Skip connection
        return out
    
class State_Encoder(nn.Module):
    "Implements State dictionary"
    def __init__(self, params):
        super(State_Encoder, self).__init__()
        self.dic_model = params.dic_model
        if params.dic_model != 0:
            self.input_layer = nn.Linear(params.pca_dim, params.dd_model)
            self.layers = nn.ModuleList([ResNetBlock(params.dd_model) for _ in range(params.N_State)])
            self.output_layer = nn.Linear(params.dd_model, params.dic_model)

    def forward(self, x):
        if self.dic_model == 0:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            return torch.cat((ones, x), dim = 1)
        else:
            y = self.input_layer(x)
            y = F.relu(y)
            for layer in self.layers:
                y = layer(y)
            y = self.output_layer(y)
            ones = torch.ones(x.shape[0], 1).to(x.device)
            y = torch.cat((ones, x, y), dim = 1)
            return y
        
class Matrix_NN(nn.Module):
    def __init__(self, params):
        super(Matrix_NN, self).__init__()
        self.input_layer = nn.Linear(params.u_dim, params.Ku_ff)
        self.k_size = params.d_model
        self.layers = nn.ModuleList([ResNetBlock(params.Ku_ff) for _ in range(params.N_Matrix)])
        self.output_layer = nn.Linear(params.Ku_ff, self.k_size * self.k_size)
    
    def forward(self, u):
        x = self.input_layer(u)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.reshape(-1, self.k_size, self.k_size)
        return x

class PCAKoopmanWithInputsInMatrix(nn.Module):
    def __init__(self, params, minmax_layer_1, pca_transformer, minmax_layer_2, minmax_layer_u, state_dic, state_matrix):
        super(PCAKoopmanWithInputsInMatrix, self).__init__()
        self.params = params
        self.minmax_layer_1 = minmax_layer_1
        self.minmax_layer_u = minmax_layer_u
        self.pca_transformer = pca_transformer
        self.minmax_layer_2 = minmax_layer_2
        self.state_dic = state_dic
        self.state_matrix = state_matrix
    
    def forward(self, x, u):
        x_pca_rescaled = self.encode(x)
        y_pca_scaled = self.pca_forward(x_pca_rescaled, u)
        y = self.pca_decode(y_pca_scaled)
        return y
        
    
    def pca_forward(self, x_pca_scaled, u):
        x_psi = self.state_dic(x_pca_scaled)
        u = self.minmax_layer_u.transform(u)
        K = self.state_matrix(u)
        x_psi_extended = x_psi.unsqueeze(1)
        y_psi = torch.matmul(x_psi_extended, K).squeeze(1)
        y_pca = y_psi[:, 1:self.params.pca_dim+1]
        return y_pca

    
    def latent_to_latent_forward(self, x_psi, u):
        u = self.minmax_layer_u.transform(u)
        K = self.state_matrix(u)
        x_psi_extended = x_psi.unsqueeze(1)
        y_psi = torch.matmul(x_psi_extended, K).squeeze(1)
        return y_psi
    
    def dic(self, x):
        x = self.encode(x)
        x = self.state_dic(x)
        return x

    def encode(self, x):
        x = self.minmax_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.minmax_layer_2.transform(x)
        return x
    
    def decode(self, psi_x):
        x = psi_x[:, 1:self.params.pca_dim+1]
        x = self.pca_decode(x)
        return x
    
    def pca_decode(self, x_pca):
        x = self.minmax_layer_2.inverse_transform(x_pca)
        x = self.pca_transformer.inverse_transform(x)
        x = self.minmax_layer_1.inverse_transform(x)
        return x
    
def build_model_MatrixWithInputs(params, x_data, u_data):
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor

    # minmax Layer 1
    min_val_1, _ = torch.min(x_data, dim=0)
    max_val_1, _ = torch.max(x_data, dim=0)
    # print(min_val_1, max_val_1)
    minmax_layer_1 = MinMaxScalerLayer(min_val_1, max_val_1)

    # rescale x_data
    x_data_scaled = minmax_layer_1.transform(x_data)

    # minmax Layer u
    min_val_u, _ = torch.min(u_data, dim=0)
    max_val_u, _ = torch.max(u_data, dim=0)
    minmax_layer_u = MinMaxScalerLayer(min_val_u, max_val_u)

    # PCA layer
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # minmax Layer 2
    min_val_2, _ = torch.min(x_pca_tensor, dim=0)
    max_val_2, _ = torch.max(x_pca_tensor, dim=0)
    minmax_layer_2 = MinMaxScalerLayer(min_val_2, max_val_2)

    # State dictionary
    state_dic = State_Encoder(params)

    # State Matrix
    state_matrix = Matrix_NN(params)

    model = PCAKoopmanWithInputsInMatrix(params, minmax_layer_1, pca_layer, minmax_layer_2, minmax_layer_u, state_dic, state_matrix)
    return model

class Params:
    def __init__(self, x_dim, u_dim, config):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.pca_dim = config.get('pca_dim', 4)
        self.dic_model = config.get('dic_model', 20)
        self.d_model = 1+ self.pca_dim + self.dic_model
        self.dd_model = config.get('dd_model', 64)
        self.dropout = config.get('dropout', 0.1)
        self.N_State = config.get('N_State', 6)
        self.Ku_ff = config.get('Ku_ff', 256)
        self.N_Matrix = config.get('N_Matrix', 6)