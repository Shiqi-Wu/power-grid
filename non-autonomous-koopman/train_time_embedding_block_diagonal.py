import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
torch.set_default_dtype(torch.float64)
from model import TimeEmbeddingBlockDiagonalKoopman


def loss_fn_multi_step(model, y, criterion, device):
    y_pred = torch.zeros_like(y).to(device)
    y_pred[:,0,:] = y[:,0,:]
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
        x_dic = model.dictionary(x)
        y_dic = model.dictionary(y)
        y_pred = model(x_dic)
        loss = criterion(y_pred, y_dic)
        loss.backward()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def test_one_epoch_single_step(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x, y = x.to(device), y.to(device)
            x_dic = model.dictionary(x)
            y_dic = model.dictionary(y)
            y_pred = model(x_dic)
            loss = criterion(y_pred, y_dic)
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
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config['step_size'], gamma = config['gamma'])
    train_losses = []
    test_losses = []
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch_single_step(model, optimizer, criterion, train_loader, device)
        test_loss = test_one_epoch_single_step(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses

def train_multi_step(model, config, device, train_loader, test_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config['step_size'], gamma = config['gamma'])
    train_losses = []
    test_losses = []
    for epoch in range(config['epochs']):
        train_loss = train_one_epoch_multi_step(model, optimizer, criterion, train_loader, device)
        test_loss = test_one_epoch_multi_step(model, criterion, test_loader, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    return train_losses, test_losses

def train(model, config, device, train_loader, test_loader):
    if config['multi_step']:
        return train_multi_step(model, config, device, train_loader, test_loader)
    else:
        return train_single_step(model, config, device, train_loader, test_loader)
    
