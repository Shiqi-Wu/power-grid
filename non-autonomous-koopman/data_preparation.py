import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from pca_layers import PCALayer, StdScalerLayer, Rescale_pca_layer

torch.set_default_dtype(torch.float64)

def data_delete_columns(x_dataset, columns = [9, 21, 25, 39, 63]):
    """
    Remove specified columns from each array in the dataset.
    Parameters:
    x_dataset (list of np.ndarray): A list of 2D numpy arrays from which columns will be removed.
    columns (list of int, optional): A list of column indices to be removed from each array. 
                                     Default is [9, 39, 63].
    Returns:
    list of np.ndarray: A new list of 2D numpy arrays with the specified columns removed.
    """

    x_dataset_new = []
    for x_data in x_dataset:
        x_data_new = np.delete(x_data, columns, axis=1)
        x_dataset_new.append(x_data_new)
    return x_dataset_new

def data_preprocessing(x_dataset, u_dataset, pca_dim):
    """
    Preprocesses the input datasets by normalizing and applying PCA transformation.
    This function performs the following steps:
    1. Concatenates the input datasets along the first axis.
    2. Normalizes the concatenated datasets using their respective means and standard deviations.
    3. Applies PCA transformation to the normalized x_data.
    4. Creates scaling layers for the normalized and PCA-transformed data.
    5. Returns a rescaled PCA layer that combines all the scaling and PCA transformations.
    Args:
        x_dataset (list of np.ndarray): List of arrays containing the x data to be concatenated and processed.
        u_dataset (list of np.ndarray): List of arrays containing the u data to be concatenated and processed.
        pca_dim (int): The number of principal components to keep during PCA transformation.
    Returns:
        Rescale_pca_layer: An object that encapsulates the scaling and PCA transformations.
    """

    x_data = np.concatenate(x_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    mean_1 = np.mean(x_data, axis=0)
    std_1 = np.std(x_data, axis=0)
    std_scaler_1 = StdScalerLayer(mean_1, std_1)
    mean_u = np.mean(u_data, axis=0)
    std_u = np.std(u_data, axis=0)
    std_scaler_u = StdScalerLayer(mean_u, std_u)
    x_data = (x_data - mean_1) / std_1
    u_data = (u_data - mean_u) / std_u
    pca = PCA(n_components=pca_dim)
    pca.fit(x_data)
    pca_matrix = torch.tensor(pca.components_, dtype=torch.float64)
    pca_layer = PCALayer(x_data.shape[1], pca_dim, pca_matrix)
    x_data = pca.transform(x_data)
    mean_2 = np.mean(x_data, axis=0)
    std_2 = np.std(x_data, axis=0)
    std_scaler_2 = StdScalerLayer(mean_2, std_2)

    rescale_pca_layer = Rescale_pca_layer(std_scaler_1, std_scaler_2, std_scaler_u, pca_layer)

    x_dataset_new = []
    u_dataset_new = []
    for x, u in zip(x_dataset, u_dataset):
        x = (x - mean_1) / std_1
        x = pca.transform(x)
        x = (x - mean_2) / std_2
        u = (u - mean_u) / std_u
        x_dataset_new.append(x)
        u_dataset_new.append(u)

        

    return rescale_pca_layer, x_dataset_new, u_dataset_new
    

def build_time_embedding_train_test_dataloader(x_dataset, u_dataset, N = 1, batch_size = 512, test_size = 0.2):
    """
    Handles both single and multiple trajectory datasets.
    """
    if len(x_dataset) == 1:
        # Single trajectory: split along time axis
        x_data = x_dataset[0]
        u_data = u_dataset[0]
        T = x_data.shape[0]
        split = int(T * (1 - test_size))

        x_train = [x_data[:split]]
        u_train = [u_data[:split]]
        x_test = [x_data[split - N:]]  # 保留 N 步历史
        u_test = [u_data[split - N:]]

    else:
        # Multiple trajectories: split by samples
        indices = np.arange(len(x_dataset))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_size))
        train_indices, test_indices = indices[:split], indices[split:]

        x_train = [x_dataset[i] for i in train_indices]
        x_test = [x_dataset[i] for i in test_indices]
        u_train = [u_dataset[i] for i in train_indices]
        u_test = [u_dataset[i] for i in test_indices]

    # 构造 time-embedding 数据集
    train_dataset = build_time_embedding_dataset(x_train, u_train, N)
    test_dataset = build_time_embedding_dataset(x_test, u_test, N)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_time_embedding_dataset(x_dataset, u_dataset, N=1):
    """
    Builds a time-embedded dataset from the given input and control datasets.
    Args:
        x_dataset (list of np.ndarray or torch.Tensor): List of input data arrays, each with shape (T, n_x).
        u_dataset (list of np.ndarray or torch.Tensor): List of control data arrays, each with shape (T, n_u).
        N (int, optional): The embedding dimension, i.e., the number of past time steps to include. Default is 1.
    Returns:
        torch.utils.data.TensorDataset: A PyTorch dataset containing the time-embedded input, output, and control data.
    Note:
        Assumes the lengths of `x_dataset` and `u_dataset` are the same and their corresponding elements have
        matching lengths along the time dimension (T).
    """
    x_dataset_time_embedded = []
    u_dataset_time_embedded = []
    y_dataset_time_embedded = []

    # Iterate over pairs of x_data and u_data
    for x_data, u_data in zip(x_dataset, u_dataset):
        # Ensure the data is a NumPy array or PyTorch tensor
        if not isinstance(x_data, (np.ndarray, torch.Tensor)):
            raise TypeError(f"x_data must be a np.ndarray or torch.Tensor, but got {type(x_data)}")
        if not isinstance(u_data, (np.ndarray, torch.Tensor)):
            raise TypeError(f"u_data must be a np.ndarray or torch.Tensor, but got {type(u_data)}")

        # Convert to PyTorch tensor if needed
        if isinstance(x_data, np.ndarray):
            x_data = torch.tensor(x_data)
        if isinstance(u_data, np.ndarray):
            u_data = torch.tensor(u_data)

        # Check dimensions
        if x_data.dim() != 2 or u_data.dim() != 2:
            raise ValueError("x_data and u_data must be 2D tensors with shapes (T, n_x) and (T, n_u), respectively.")

        # Create time-embedded datasets for the current sequence
        x_data_time_embedded = []
        u_data_time_embedded = []
        y_data_time_embedded = []
        for i in range(N, x_data.shape[0] - 1):
            x_data_time_embedded.append(x_data[i - N:i, :].reshape(1, -1))
            y_data_time_embedded.append(x_data[i - N + 1:i + 1, :].reshape(1, -1))
            u_data_time_embedded.append(u_data[i - N:i, :].reshape(1, -1))

        # Concatenate the embeddings along the batch dimension
        x_data_time_embedded = torch.cat(x_data_time_embedded, dim=0)
        u_data_time_embedded = torch.cat(u_data_time_embedded, dim=0)
        y_data_time_embedded = torch.cat(y_data_time_embedded, dim=0)

        # Append to the global lists
        x_dataset_time_embedded.append(x_data_time_embedded)
        u_dataset_time_embedded.append(u_data_time_embedded)
        y_dataset_time_embedded.append(y_data_time_embedded)

    # Concatenate data from all sequences
    x_dataset_time_embedded = torch.cat(x_dataset_time_embedded, dim=0)
    u_dataset_time_embedded = torch.cat(u_dataset_time_embedded, dim=0)
    y_dataset_time_embedded = torch.cat(y_dataset_time_embedded, dim=0)

    # Create a PyTorch dataset
    dataset = torch.utils.data.TensorDataset(x_dataset_time_embedded, y_dataset_time_embedded, u_dataset_time_embedded)
    return dataset


