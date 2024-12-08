import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
torch.set_default_dtype(torch.float64)
from model import TimeEmbeddingBlockDiagonalKoopman
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from args_arguments import parse_arguments, read_config_file
from load_dataset import load_dataset_from_files
from data_preparation import data_delete_columns, data_preprocessing, build_time_embedding_train_test_dataloader
from tqdm import tqdm

def loss_fn_multi_step(model, y, criterion, device):
    y_pred = torch.zeros_like(y).to(device)  # 确保 y_pred 在目标设备上
    y_pred[:, 0, :] = y[:, 0, :]
    for i in range(1, y.shape[1]):
        y_pred[:, i] = model(y_pred)[:, 0]
    loss = criterion(y_pred, y)
    return loss

def train_one_epoch_single_step(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for x, y, _ in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model.dictionary_forward(x)
        y_true = model.dictionary_V(y)
        loss = criterion(y_pred, y_true) + model.regularization_loss()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def test_one_epoch_single_step(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model.dictionary_forward(x)
            y_true = model.dictionary_V(y)
            loss = criterion(y_pred, y_true) + model.regularization_loss()
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_one_epoch_multi_step(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for y, _ in train_loader:
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fn_multi_step(model, y, criterion, device)
        loss.backward()
        optimizer.step()  # 更新参数
        train_loss += loss.item()
    return train_loss / len(train_loader)

def test_one_epoch_multi_step(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for y, _ in test_loader:
            y = y.to(device)
            loss = loss_fn_multi_step(model, y, criterion, device)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_single_step(model, config, device, train_loader, test_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(config['epochs']), desc="Training", unit="epoch"):
        train_loss = train_one_epoch_single_step(model, optimizer, criterion, train_loader, device)
        test_loss = test_one_epoch_single_step(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
        tqdm.write(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses

def train_multi_step(model, config, device, train_loader, test_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    train_losses = []
    test_losses = []
    for epoch in tqdm(range(config['epochs']), desc="Training", unit="epoch"):
        train_loss = train_one_epoch_multi_step(model, optimizer, criterion, train_loader, device)
        test_loss = test_one_epoch_multi_step(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
        tqdm.write(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses


def train(model, config, device, train_loader, test_loader):
    if config['multi_step']:
        return train_multi_step(model, config, device, train_loader, test_loader)
    else:
        return train_single_step(model, config, device, train_loader, test_loader)
    
def main_single_step():
    # Parse command-line arguments and read configuration file
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    x_dataset, u_dataset = load_dataset_from_files(config)

    x_dataset = data_delete_columns(x_dataset)
    rescale_pca_layer, x_dataset_new, u_dataset_new = data_preprocessing(x_dataset, u_dataset, config['pca_dim'])

    print(f"Number of samples: {len(x_dataset_new)}")

    train_loader, test_loader = build_time_embedding_train_test_dataloader(x_dataset_new, u_dataset_new, N = config['N'], batch_size = config['batch_size'], test_size = config['test_size'])
    print(f"data loaded")

    # Build model
    model = TimeEmbeddingBlockDiagonalKoopman(dictionary_dim = config['dictionary_dim'], inputs_dim = config['N'] * config['pca_dim'], num_blocks = config['num_blocks'])
    model.to(device)

    # for name, param in model.named_parameters():
    #     print(f"Parameter {name} is on {param.device}")


    # Train model
    train_losses, test_losses = train(model, config, device, train_loader, test_loader)

    # Save model
    model_path = os.path.join(config['save_dir'], 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    std_pca_layer_path = os.path.join(config['save_dir'], 'std_pca_layer.pth')
    torch.save(rescale_pca_layer, std_pca_layer_path)
    print(f"Standardization and PCA layer saved to {std_pca_layer_path}")

    # Save losses
    losses = {'train_losses': train_losses, 'test_losses': test_losses}
    losses_path = os.path.join(config['save_dir'], 'losses.pth')
    torch.save(losses, losses_path)
    print(f"Losses saved to {losses_path}")

    return model, train_losses, test_losses

if __name__ == '__main__':
    main_single_step()



