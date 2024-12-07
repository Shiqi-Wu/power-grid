import torch
import torch.nn as nn

class PCALayer(nn.Module):
    """
    A PyTorch layer that performs Principal Component Analysis (PCA) transformation.
    Args:
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data after PCA transformation.
        pca_matrix (torch.Tensor): The PCA transformation matrix.
    Attributes:
        pca_matrix (torch.Tensor): The PCA transformation matrix.
        input_dim (int): The dimensionality of the input data.
        output_dim (int): The dimensionality of the output data after PCA transformation.
        transform (nn.Linear): A linear layer that applies the PCA transformation.
        inverse_transform (nn.Linear): A linear layer that applies the inverse PCA transformation.
    """

    def __init__(self, input_dim, output_dim, pca_matrix):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.transform.weight = nn.Parameter(pca_matrix, requires_grad=False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        self.inverse_transform.weight = nn.Parameter(pca_matrix.T, requires_grad=False)

class StdScalerLayer(nn.Module):
    def __init__(self, mean, std):
        super(StdScalerLayer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, input):
        return input * self.std + self.mean

class Rescale_pca_layer(nn.Module):
    def __init__(self, std_layer_1, std_layer_2, std_layer_u, pca_layer):
        super(Rescale_pca_layer, self).__init__()
        self.std_layer_1 = std_layer_1
        self.std_layer_2 = std_layer_2
        self.std_layer_u = std_layer_u
        self.pca_layer = pca_layer

    def transform_x(self, x):
        x = self.std_layer_1.transform(x)
        x = self.pca_layer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def inverse_transform_x(self, x):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_layer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x

    def transform_u(self, u):
        return self.std_layer_u.transform(u)

    def inverse_transform_u(self, u):
        return self.std_layer_u.inverse_transform(u)