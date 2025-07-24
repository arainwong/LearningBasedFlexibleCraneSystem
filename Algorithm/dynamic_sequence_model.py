from utils import *
from models.dynamic_sequence_NN import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import wandb

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')

    config = load_config('Algorithm/configs/config.yaml')

    # basic setting configurations
    seed = config['config']['seed']
    data_config = config['config']['data_config']
    target_config = config['config']['target_config']
    nn_config = config['config']['nn_config']

    wandb_enable = config['wandb']['wandb_enable']

    dataset_folder = config['data_setting']['dataset_folder']
    command_type_set = config['data_setting']['command_type_set']
    add_noise = config['data_setting']['add_noise']

    # model configurations
    input_horizon = config['model']['input_horizon']
    output_horizon = config['model']['output_horizon']
    hidden_dim = config['model']['hidden_dim']
    saved_path = config['model']['saved_path']

    # training configurations
    data_normalization = config['training']['data_normalization']
    epochs = config['training']['epochs']
    eval_freq = config['training']['eval_freq']

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = load_dataset(dataset_folder, command_type_set)
    # show_dataset_shape(dataset)

    static_inputs = prepare_static_dataset(dataset)
    seq_inputs, targets  = prepare_dynamic_sequence_dataset(dataset, input_horizon, output_horizon, data_config, target_config, add_noise=add_noise)
    if data_normalization:
        dynamic_sequence_inputs_normalizer = Normalizer()
        dynamic_sequence_inputs_normalizer.fit(seq_inputs)
        seq_inputs = dynamic_sequence_inputs_normalizer.normalize(seq_inputs)
        print(f'Daynamic sequence inputs have been normalized.')

        targets_normalizer = Normalizer()
        targets_normalizer.fit(targets)
        targets = targets_normalizer.normalize(targets)
        print(f'Daynamic sequence targets have been normalized. \nTarget mean: {targets_normalizer.mean} \nTarget std: {targets_normalizer.std} \n')
    else:
        print(f'Data normalization disabled. \n')
    print(f'Static inputs shape: {static_inputs.shape} \nDynamic sequence inputs shape: {seq_inputs.shape} \nTargets shape: {targets.shape} \n')
    static_dim = static_inputs.shape[0]
    seq_feature_dim = seq_inputs.shape[2]
    _, output_horizon, output_dim = targets.shape

    if nn_config == 'LSTM':
        full_dataset = DynamicsLSTMDataset(static_inputs, seq_inputs, targets)
        dynamic_model = DynamicsLSTM(static_dim=static_dim, seq_feature_dim=seq_feature_dim, 
                                     output_horizon=output_horizon, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
    elif nn_config == 'GRU':
        full_dataset = DynamicsGRUDataset(static_inputs, seq_inputs, targets)
        dynamic_model = DynamicsGRU(static_dim=static_dim, seq_feature_dim=seq_feature_dim, 
                                     output_horizon=output_horizon, output_dim=output_dim, hidden_dim=hidden_dim).to(device)
    elif nn_config == 'Transformer':
        full_dataset = DynamicsTransformerDataset(static_inputs, seq_inputs, targets)
        dynamic_model = DynamicsTransformer(static_dim=static_dim, seq_feature_dim=seq_feature_dim, 
                                     output_horizon=output_horizon, output_dim=output_dim, hidden_dim=hidden_dim, max_seq_len=input_horizon).to(device)
    print(f'Using model type: {nn_config} \n')

    train_proportion = config['training']['train_proportion']
    batch_size = config['training']['batch_size']

    train_size = int(train_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # print(f'{dataloader}: {len(dataloader)}')

    if wandb_enable:
        wandb.init(project='dynamics_model', name=f'{nn_config}_inH_{input_horizon}_outH_{output_horizon}_{data_config}_{target_config}')
    
    if data_normalization:
        dynamic_sequence_inputs_normalizer.numpy_to_tensor()
        targets_normalizer.numpy_to_tensor()
    torch.autograd.set_detect_anomaly(True)
    min_test_loss = float('inf')
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_static_inputs, batch_seq_inputs, batch_targets in train_dataloader:
            batch_static_inputs = batch_static_inputs.to(device)
            batch_seq_inputs = batch_seq_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = dynamic_model(batch_static_inputs, batch_seq_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamic_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        if wandb_enable:
            wandb.log({"train_loss": avg_loss}, step=epoch)

        # evaluation
        if epoch % eval_freq == 0:
            dynamic_model.eval()
            test_loss = 0.0
            denormalized_test_loss = 0.0
            mean_error = torch.zeros(output_horizon * output_dim, device=device)
            with torch.no_grad():
                for test_static_inputs, test_seq_inputs, test_targets in test_dataloader:
                    test_static_inputs = test_static_inputs.to(device)
                    test_seq_inputs = test_seq_inputs.to(device)
                    test_targets = test_targets.to(device)
                    
                    test_outputs = dynamic_model(test_static_inputs, test_seq_inputs)
                    
                    batch_loss = criterion(test_outputs, test_targets).item()
                    test_loss += batch_loss

                    if data_normalization:
                        test_targets_real = targets_normalizer.denormalize(test_targets)
                        test_outputs_real = targets_normalizer.denormalize(test_outputs)

                        denormalized_batch_loss = criterion(test_outputs_real, test_targets_real).item()
                        denormalized_test_loss += denormalized_batch_loss
                    else:
                        test_targets_real = test_targets
                        test_outputs_real = test_outputs

                    error = (test_targets_real - test_outputs_real).abs()
                    batch_mean_error = error.mean(dim=(0)).flatten()
                    mean_error += batch_mean_error

            avg_test_loss = test_loss / len(test_dataloader)
            if data_normalization:
                avg_denormalized_test_loss = denormalized_test_loss / len(test_dataloader)
                print(f'Epoch: {epoch+1}, test loss: {avg_test_loss:.6f}; denormalized test loss: {avg_denormalized_test_loss:.6f}')
            else:
                print(f'Epoch: {epoch+1}, test loss: {avg_test_loss:.6f}')
            
            # save dynamic model once get lowest test loss
            if data_normalization and min_test_loss > avg_denormalized_test_loss:
                min_test_loss = avg_denormalized_test_loss
                torch.save(dynamic_model.state_dict(), f'{saved_path}/dynamic_{nn_config}_inH_{input_horizon}_outH_{output_horizon}_{data_config}_{target_config}.pth')
                print(f'New lowest denormalized test loss {min_test_loss:.6f}, dynamic model saved in path: {saved_path}.')
            elif (not data_normalization) and min_test_loss > avg_test_loss:
                min_test_loss = avg_test_loss
                torch.save(dynamic_model.state_dict(), f'{saved_path}/dynamic_{nn_config}_inH_{input_horizon}_outH_{output_horizon}_{data_config}_{target_config}.pth')
                print(f'New lowest test loss {min_test_loss:.6f}, dynamic model saved in path: {saved_path}.')

            test_target = test_targets_real[0, :, 0:output_dim].detach().cpu()
            test_output = test_outputs_real[0, :, 0:output_dim].detach().cpu()
            print(f'target: {test_target}, \ntest output: {test_output}')
            avg_mean_error = (mean_error / len(test_dataloader)).detach().cpu()
            print(f'Target features average error: {avg_mean_error}')
            
            if wandb_enable:
                wandb.log({"test_loss": avg_test_loss}, step=epoch)
                if data_normalization:
                    wandb.log({"denormalized_test_loss": avg_denormalized_test_loss}, step=epoch)
                wandb.log({
                    "test_loss": avg_test_loss,
                    **{f"target_feature_error_{i+1}": v.item() for i, v in enumerate(avg_mean_error[0:output_dim])}
                }, step=epoch)
            
            dynamic_model.train()

    if wandb_enable:
        wandb.finish()
    