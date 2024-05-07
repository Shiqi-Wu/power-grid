from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/shiqi/code/Project4-power-grid/utils')
sys.path.append('/home/shiqi/code/Project4-power-grid/pca_koopman')
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import pca_koopman_dir as km
from load_dataset import *
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def koopman_loss(model, x, u):
    loss_fn = nn.MSELoss()
    N = x.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    x_latent = model.encode(x)
    x0 = model.state_dic(x_latent[:, 0, :])
    x_pred = torch.zeros((x_latent.shape[0], x_latent.shape[1], x0.shape[1]), dtype=torch.float32, device=x.device)
    x_true = torch.zeros((x_latent.shape[0], x_latent.shape[1], x0.shape[1]), dtype=torch.float32, device=x.device)
    
    x_pred[:, 0, :] = x0
    x_true[:, 0, :] = x0
    for i in range(1, N):
        x_pred_cur = model.latent_to_latent_forward(x0, u[:, i, :])
        x_pred[:, i, :] = x_pred_cur
        x_true[:, i, :] = model.state_dic(x_latent[:, i, :])
    loss = loss_fn(x_pred, x_true)
    return loss

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    i = 0
    for x, u in train_loader:
        # i += 1
        x, u = x.to(device), u.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, u)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        # print("Batch: ", i, "Loss: ", loss.item())
    return total_loss/len(train_loader)

def test_one_epoch(model, test_loader,loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, u in test_loader:
            x, u = x.to(device), u.to(device)
            loss = loss_fn(model, x, u)
            total_loss += loss.item()
    return total_loss/len(test_loader)

def main(config):

    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # build model
    x_dataset, u_dataset = load_dataset_from_files(config)
    x_data = np.concatenate(x_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    x_dim = np.shape(x_data)[1]
    u_dim = np.shape(u_data)[1]
    params = km.Params(x_dim=x_dim, u_dim=u_dim, config = config)
    model = km.build_model_MatrixWithInputs(params, x_data, u_data)
    model.to(device)

    # Load dataset
    x_data_slices, u_data_slices = build_training_dataset(config, x_dataset, u_dataset)

    # Split the data
    x_train, x_test, u_train, u_test = train_test_split(x_data_slices, u_data_slices, test_size=0.2, random_state=42)

    # Convert numpy arrays back to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    u_test = torch.tensor(u_test, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(x_train, u_train)
    test_dataset = TensorDataset(x_test, u_test)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Loss function
    loss_fn = koopman_loss

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))
    
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)

    return

if __name__ == '__main__':
    args = parse_arguments()
    config = read_config_file(args.config)
    main(config)
