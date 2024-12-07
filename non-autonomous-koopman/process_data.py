import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
torch.set_default_dtype(torch.float64)

def data_preprocessing(x_data, y_data, u_data, pca_dim):
    

def build_time_embedding_train_test_dataloader(x_dataset, u_dataset, N = 1, batch_size = 64, test_size = 0.2):
    """
    Splits the given datasets into training and testing sets, applies time embedding, and returns DataLoader objects for both sets.
    Args:
        x_dataset (list or np.ndarray): The dataset containing the state variables.
        u_dataset (list or np.ndarray): The dataset containing the control inputs.
        N (int, optional): The time embedding dimension. Defaults to 1.
        batch_size (int, optional): The number of samples per batch to load. Defaults to 64.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing set.
    """

    indices = np.arange(len(x_dataset))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_indices, test_indices = indices[:split], indices[split:]

    x_train = [x_dataset[i] for i in train_indices]
    x_test = [x_dataset[i] for i in test_indices]
    u_train = [u_dataset[i] for i in train_indices]
    u_test = [u_dataset[i] for i in test_indices]

    train_dataset = build_time_embedding_dataset(x_train, u_train, N)
    test_dataset = build_time_embedding_dataset(x_test, u_test, N)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def build_time_embedding_dataset(x_dataset, u_dataset, N = 1):
    """
        Builds a time-embedded dataset from the given input and control datasets.
        Args:
            x_dataset (list of np.ndarray): List of input data arrays, where each array has shape (T, n_x).
            u_dataset (list of np.ndarray): List of control data arrays, where each array has shape (T, n_u).
            N (int, optional): The embedding dimension, i.e., the number of past time steps to include. Default is 1.
        Returns:
            torch.utils.data.TensorDataset: A PyTorch dataset containing the time-embedded input, output, and control data.
                - The first tensor in the dataset is the time-embedded input data.
                - The second tensor in the dataset is the time-embedded output data.
                - The third tensor in the dataset is the time-embedded control data.
        Note:
            The function assumes that the length of x_dataset and u_dataset are the same and that each corresponding pair
            of elements in these lists have the same length along the time dimension (T).
    """

    for x_data, u_data in zip(x_dataset, u_dataset):
        x_data_time_embedded = []
        u_data_time_embedded = []
        y_data_time_embedded = []
        for i in range(N, len(x_data)-1):
            x_data_time_embedded.append(x_data[i-N:i, :].view(1, -1))
            y_data_time_embedded.append(x_data[i-N+1: i+1, :].view(1, -1))
            u_data_time_embedded.append(u_data[i-N:i, :].view(1, -1))

        x_data_time_embedded = np.concatenate(x_data_time_embedded, axis=0)
        u_data_time_embedded = np.concatenate(u_data_time_embedded, axis=0)
        y_data_time_embedded = np.concatenate(y_data_time_embedded, axis=0)

        x_dataset_time_embedded.append(x_data_time_embedded)
        u_dataset_time_embedded.append(u_data_time_embedded)
        y_dataset_time_embedded.append(y_data_time_embedded)
    
    x_dataset_time_embedded = np.concatenate(x_dataset_time_embedded, axis=0)
    u_dataset_time_embedded = np.concatenate(u_dataset_time_embedded, axis=0)
    y_dataset_time_embedded = np.concatenate(y_dataset_time_embedded, axis=0)
    x_dataset_time_embedded = torch.tensor(x_dataset_time_embedded)
    u_dataset_time_embedded = torch.tensor(u_dataset_time_embedded)
    y_dataset_time_embedded = torch.tensor(y_dataset_time_embedded)

    dataset = torch.utils.data.TensorDataset(x_dataset_time_embedded, y_dataset_time_embedded, u_dataset_time_embedded)
    return dataset

