import torch
import numpy as np
import os
torch.set_default_dtype(torch.float64)
from model import TimeEmbeddingBlockDiagonalKoopman, KoopmanModelWithInputs
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from args_arguments import parse_arguments, read_config_file
from load_dataset import load_dataset_from_files
from data_preparation import data_delete_columns, data_preprocessing, build_time_embedding_train_test_dataloader
from tqdm import tqdm
from pca_layers import StdScalerLayer
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    # batch 是一个长度为 1 的列表，包含单个 mini-batch tuple
    return batch[0]  # 返回 tuple，而不是 [tuple]


def rescale_residuals(residuals):
    all_residuals = torch.cat([res for _, res, _ in residuals], dim=0)
    mean = all_residuals.mean(dim=0, keepdim=True)
    std = all_residuals.std(dim=0, keepdim=True)
    rescaled_residuals = [(x, (res - mean) / std, u) for x, res, u in residuals]
    return rescaled_residuals, mean, std


def build_residual_dataset_model1(dataloader, model1, device, rescale = True):
    residuals = []
    for x, y, u in dataloader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x).detach()
        y_pred = model1.dictionary_forward(x)
        y_true = model1.dictionary_V(y)
        residual = y_true - y_pred
        residual = residual.detach()
        # print(f"x_dic shape: {x_dic.shape}, residual shape: {residual.shape}, u shape: {u.shape}")
        residuals.append((x_dic, residual, u))

    residuals, mean, std = rescale_residuals(residuals)
    new_dataloader = torch.utils.data.DataLoader(
    residuals,
    batch_size=1,
    collate_fn=custom_collate_fn
)
    if not rescale:
        mean = torch.zeros_like(mean)
        std = torch.ones_like(std)

    std_layer_residuals = StdScalerLayer(mean, std)

    return new_dataloader, std_layer_residuals
    
def build_residual_dataset_model2(dataloader, model2, std_layer_residuals, device):
    residuals = []
    for x, y, u in dataloader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model2.blockdiagonalkoopmanmodel.dictionary_V(x).detach()
        y_dic_pred_rescaled = model2(x_dic, u)
        y_dic_pred = std_layer_residuals.inverse_transform(y_dic_pred_rescaled)
        y_dic = model2.blockdiagonalkoopmanmodel.dictionary_V(y)
        residual = y_dic - y_dic_pred
        residual = residual.detach()
        residuals.append((x_dic, residual, u))
    new_dataloader = torch.utils.data.DataLoader(
    residuals,
    batch_size=1,
    collate_fn=custom_collate_fn
)
    return new_dataloader

def train_one_epoch_blockdiagonal(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for x, y, u in train_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def test_one_epoch_blockdiagonal(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, u in test_loader:
            x, y, u = x.to(device), y.to(device), u.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_one_epoch_residual(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for x, y, u in train_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        optimizer.zero_grad()
        y_pred = model(x, u)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def test_one_epoch_residual(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y, u in test_loader:
            x, y, u = x.to(device), y.to(device), u.to(device)
            y_pred = model(x, u)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
    return test_loss / len(test_loader)

def train_blockdiagonal_model(model1, model2, std_layer_residuals, config, train_loader, test_loader, device):
    train_loader_new = build_residual_dataset_model2(train_loader, model2, std_layer_residuals, device)
    test_loader_new = build_residual_dataset_model2(test_loader, model2, std_layer_residuals, device)
    optimizer = torch.optim.Adam(model1.parameters(), lr=config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    criterion = torch.nn.MSELoss()


    train_losses = []
    test_losses = []
    for epoch in tqdm(range(config['epochs']), desc="Training", unit="epoch"):
        train_loss = train_one_epoch_blockdiagonal(model1, optimizer, criterion, train_loader_new, device)
        test_loss = test_one_epoch_blockdiagonal(model1, criterion, test_loader_new, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
    tqdm.write(f"Final train loss: {train_losses[-1]:.4e}")
    return train_losses, test_losses

def train_residual_model(model1, model2, config, train_loader, test_loader, device):
    train_loader_new, std_layer_residuals = build_residual_dataset_model1(train_loader, model1, device, rescale = config['rescale_residuals'])
    test_loader_new, _ = build_residual_dataset_model1(test_loader, model1, device, rescale = config['rescale_residuals'])
    optimizer = torch.optim.Adam(model2.parameters(), lr=config['lr'])
    stepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])
    criterion = torch.nn.MSELoss()

    train_losses = []
    test_losses = []
    for epoch in tqdm(range(config['epochs']), desc="Training", unit="epoch"):
        train_loss = train_one_epoch_residual(model2, optimizer, criterion, train_loader_new, device)
        test_loss = test_one_epoch_residual(model2, criterion, test_loader_new, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        stepLR.step()
    tqdm.write(f"Final train loss: {train_losses[-1]:.4e}")
    return train_losses, test_losses, std_layer_residuals

def evaluate_one_step_pred(model1, model2, std_layer_residuals, config, train_loader, test_loader, device):
    V_inv = torch.inverse(model1.koopman.V)
    train_eval_loss = 0.0
    for x, y, u in train_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x)
        y_dic_pred_1 = model1.dictionary_forward(x)
        y_dic_pred_2_rescaled = model2(x_dic, u)
        y_dic_pred_2 = std_layer_residuals.inverse_transform(y_dic_pred_2_rescaled)
        y_dic_pred = y_dic_pred_1 + y_dic_pred_2

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        train_eval_loss += loss.item()
    train_eval_loss /= len(train_loader)

    test_eval_loss = 0.0
    for x, y, u in test_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x)
        y_dic_pred_1 = model1.dictionary_forward(x)
        y_dic_pred_2_rescaled = model2(x_dic, u)
        y_dic_pred_2 = std_layer_residuals.inverse_transform(y_dic_pred_2_rescaled)
        y_dic_pred = y_dic_pred_1 + y_dic_pred_2

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        test_eval_loss += loss.item()
    test_eval_loss /= len(test_loader)
    return train_eval_loss, test_eval_loss

def evaluate_one_step_pred_initial(model1, config, train_loader, test_loader, device):
    V_inv = torch.inverse(model1.koopman.V)
    train_eval_loss = 0.0
    for x, y, u in train_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x)
        y_dic_pred_1 = model1.dictionary_forward(x)
        y_dic_pred = y_dic_pred_1

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        train_eval_loss += loss.item()
    train_eval_loss /= len(train_loader)

    test_eval_loss = 0.0
    for x, y, u in test_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x)
        y_dic_pred_1 = model1.dictionary_forward(x)
        y_dic_pred = y_dic_pred_1

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        test_eval_loss += loss.item()
    test_eval_loss /= len(test_loader)
    return train_eval_loss, test_eval_loss

def main():
    # parse commend-line arguments
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data
    x_dataset, u_dataset = load_dataset_from_files(config)
    u_data = u_dataset[0]
    u_dim = u_data.shape[1]

    # data preprocessing
    x_dataset = data_delete_columns(x_dataset)
    rescale_pca_layer, x_dataset_new, u_dataset_new = data_preprocessing(x_dataset, u_dataset, config['pca_dim'])

    print(f"Number of samples: {len(x_dataset_new)}")

    train_loader, test_loader = build_time_embedding_train_test_dataloader(x_dataset_new, u_dataset_new, 1, batch_size=config['batch_size'], test_size=config['test_size'])
    
    # Load model1
    model_blockdiagonal = TimeEmbeddingBlockDiagonalKoopman(dictionary_dim=config['dictionary_dim'], inputs_dim=config['N'] * config['pca_dim'], num_blocks=config['num_blocks'])
    model_blockdiagonal.to(device)
    model_state_dict = torch.load(config['model_blockdiagonal_path'])
    model_blockdiagonal.load_state_dict(model_state_dict)

    # Build model2
    model_residual = KoopmanModelWithInputs(koopman_dim= 1 + config['pca_dim'] + config['dictionary_dim'], u_dim=u_dim, num_blocks=config['residual_num_blocks'], blockdiagonalkoopmanmodel=model_blockdiagonal)
    model_residual.to(device)

    train_eval_losses = []
    test_eval_losses = []

    # Evaluate initial model
    train_eval_loss, test_eval_loss = evaluate_one_step_pred_initial(model_blockdiagonal, config, train_loader, test_loader, device)
    train_eval_losses.append(train_eval_loss)
    test_eval_losses.append(test_eval_loss)

    # iterative training
    for round in range(config['training_rounds']):
        train_losses, test_losses, std_layer_residuals = train_residual_model(model_blockdiagonal, model_residual, config, train_loader, test_loader, device)
        residual_losses = {'train_losses': train_losses, 'test_losses': test_losses}
        residual_losses_path = os.path.join(config['save_dir'], f'residual_losses_{round}.pth')
        torch.save(residual_losses, residual_losses_path)
        std_layer_residuals_path = os.path.join(config['save_dir'], f'std_layer_residuals_{round}.pth')
        torch.save(std_layer_residuals, std_layer_residuals_path)

        train_losses, test_losses = train_blockdiagonal_model(model_blockdiagonal, model_residual, std_layer_residuals, config, train_loader, test_loader, device)
        blockdiagonal_losses = {'train_losses': train_losses, 'test_losses': test_losses}
        blockdiagonal_losses_path = os.path.join(config['save_dir'], f'blockdiagonal_losses_{round}.pth')
        torch.save(blockdiagonal_losses, blockdiagonal_losses_path)

        train_eval_loss, test_eval_loss = evaluate_one_step_pred(model_blockdiagonal, model_residual, std_layer_residuals, config, train_loader, test_loader, device)
        train_eval_losses.append(train_eval_loss)
        test_eval_losses.append(test_eval_loss)

    iterative_losses = {'train_losses': train_eval_losses, 'test_losses': test_eval_losses}
    iterative_losses_path = os.path.join(config['save_dir'], 'iterative_losses.pth')
    torch.save(iterative_losses, iterative_losses_path)

    torch.save(model_blockdiagonal.state_dict(), os.path.join(config['save_dir'], 'model_blockdiagonal.pth'))
    torch.save(model_residual.state_dict(), os.path.join(config['save_dir'], 'model_residual.pth'))

    plt.figure()
    plt.plot(train_eval_losses, label='Train loss')
    plt.plot(test_eval_losses, label='Test loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(config['save_dir'], 'iterative_losses.png'))

if __name__ == "__main__":
    main()

        
    