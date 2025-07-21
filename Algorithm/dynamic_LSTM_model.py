from utils import *
from models.MLP import *
from models.LSTM import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import wandb

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    nn_type = 'LSTM'
    dataset_folder = 'GeneratedDataset_Train/CommandType_'
    command_type_set = ['sine', 'triangle', 'step']
    # command_type_set = ['triangle']
    dataset = load_dataset(dataset_folder, command_type_set)
    show_dataset_shape(dataset)

    input_horizon = 32
    output_horizon = 1
    train_proportion = 0.9

    if nn_type == 'LSTM':
        static_input = prepare_static_dataset(dataset)
        seq_inputs, targets  = prepare_LSTM_dataset(dataset, input_horizon, output_horizon)
        print(static_input.shape, seq_inputs.shape, targets.shape)
        full_dataset = DynamicsLSTMDataset(static_input, seq_inputs, targets)
        static_dim = static_input.shape[0]
        seq_feature_dim = seq_inputs.shape[2]
        _, output_horizon, output_dim = targets.shape
        dynamic_model = DynamicsLSTM(static_dim=static_dim, seq_feature_dim=seq_feature_dim, 
                                     output_horizon=output_horizon, output_dim=output_dim, hidden_dim=256).to(device)

    train_size = int(train_proportion * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # print(f'{dataloader}: {len(dataloader)}')

    wandb.init(project='dynamics_model', name=f'{nn_type}_training_inH_{input_horizon}_outH_{output_horizon}')
    
    epochs = 50
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
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        wandb.log({"train_loss": avg_loss}, step=epoch)

        # evaluation
        if epoch % 5 == 0:
            dynamic_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for test_static_inputs, test_seq_inputs, test_targets in test_dataloader:
                    test_static_inputs = test_static_inputs.to(device)
                    test_seq_inputs = test_seq_inputs.to(device)
                    test_targets = test_targets.to(device)
                    
                    test_outputs = dynamic_model(test_static_inputs, test_seq_inputs)
                    batch_loss = criterion(test_outputs, test_targets).item()
                    test_loss += batch_loss
            avg_test_loss = test_loss / len(test_dataloader)
            print(f'Epoch: {epoch+1}, test loss: {avg_test_loss:.6f}')
            target = test_targets[0, :, 0:3]
            test = test_outputs[0, :, 0:3]
            print(f'target: {target}, test output: {test}')

            wandb.log({"test_loss": avg_test_loss}, step=epoch)
            dynamic_model.train()

    wandb.finish()
    torch.save(dynamic_model.state_dict(), f'Algorithm/dynamic_{nn_type}_model_weights.pth')