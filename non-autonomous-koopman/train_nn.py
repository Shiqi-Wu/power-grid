import torch
import numpy as np
import os
torch.set_default_dtype(torch.float64)
from model import TimeEmbeddingBlockDiagonalKoopman, state_input_network
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


def build_residual_dataset_model1(dataloader, model1, device, sample_step = 10, rescale = True):
    residuals = []
    for x, y, u in dataloader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x).detach()
        y_pred = model1.dictionary_forward(x, sample_step)
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

def train_residual_model(model1, model2, config, train_loader, test_loader, device):
    train_loader_new, std_layer_residuals = build_residual_dataset_model1(train_loader, model1, device, sample_step=config['sample_step'], rescale = config['rescale_residuals'])
    test_loader_new, _ = build_residual_dataset_model1(test_loader, model1, device, sample_step=config['sample_step'], rescale = config['rescale_residuals'])
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
        y_dic_pred_1 = model1.dictionary_forward(x, config['sample_step'])
        y_dic_pred = y_dic_pred_1

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        train_eval_loss += loss.item()
    train_eval_loss /= len(train_loader)

    test_eval_loss = 0.0
    for x, y, u in test_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        x_dic = model1.dictionary_V(x)
        y_dic_pred_1 = model1.dictionary_forward(x, config['sample_step'])
        y_dic_pred = y_dic_pred_1

        y_pred = torch.matmul(y_dic_pred, V_inv)[:, 1:config['pca_dim']+1]
        loss = torch.nn.MSELoss()(y_pred, y)
        test_eval_loss += loss.item()
    test_eval_loss /= len(test_loader)
    return train_eval_loss, test_eval_loss

def main():
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    x_dataset, u_dataset = load_dataset_from_files(config)
    u_data = u_dataset[0]
    u_dim = u_data.shape[1]

    # data preprocessing
    x_dataset = data_delete_columns(x_dataset)
    rescale_pca_layer, x_dataset_new, u_dataset_new = data_preprocessing(x_dataset, u_dataset, config['pca_dim'])

    train_loader, test_loader = build_time_embedding_train_test_dataloader(x_dataset_new, u_dataset_new, 1, batch_size=config['batch_size'], test_size=config['test_size'])

    # Load model1
    model_blockdiagonal = TimeEmbeddingBlockDiagonalKoopman(dictionary_dim=config['dictionary_dim'], inputs_dim=config['N'] * config['pca_dim'], num_blocks=config['num_blocks'])
    model_blockdiagonal.to(device)
    model_state_dict = torch.load(config['model_blockdiagonal_path'])
    model_blockdiagonal.load_state_dict(model_state_dict)

    # Build model2
    model_residual = state_input_network(1 + config['pca_dim'] + config['dictionary_dim'], u_dim, config['residual_num_blocks'])
    model_residual.to(device)

    # train model 2
    train_losses, test_losses, std_layer_residuals = train_residual_model(model_blockdiagonal, model_residual, config, train_loader, test_loader, device)

    plt.figure()
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and testing loss')
    plt.savefig(os.path.join(config['save_dir'], 'loss_residual.png'))
    plt.show()

    # Save model
    model_path = os.path.join(config['save_dir'], 'model_residual.pth')
    torch.save(model_residual.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save losses
    losses = {'train_losses': train_losses, 'test_losses': test_losses}
    losses_path = os.path.join(config['save_dir'], 'losses_residual.pth')

    torch.save(losses, losses_path)
    print(f"Losses saved to {losses_path}")

if __name__ == '__main__':
    main()