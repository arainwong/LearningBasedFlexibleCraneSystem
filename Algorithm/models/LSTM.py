import torch
import torch.nn as nn
from torch.utils.data import Dataset

class DynamicsLSTMDataset(Dataset):
    def __init__(self, static_inputs, seq_inputs, targets):
        self.static_inputs = torch.from_numpy(static_inputs).float()
        self.seq_inputs = torch.from_numpy(seq_inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return self.seq_inputs.shape[0]

    def __getitem__(self, idx):
        return self.static_inputs, self.seq_inputs[idx], self.targets[idx]
    
class DynamicsLSTM(nn.Module):
    def __init__(self, static_dim, seq_feature_dim, output_horizon, output_dim, hidden_dim=128, lstm_layers=2):
        super().__init__()
        self.output_horizon = output_horizon
        self.output_dim = output_dim

        # sys and body features
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # dynamic sequence encoder（or GRU/Transformer）
        self.dynamic_encoder = nn.LSTM(
            input_size=seq_feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )

        # fuse data to predict
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, static_input, sequence_input):
        # static encoder
        static_feat = self.static_encoder(static_input)  # [batch_size, hidden_dim]

        # dynamic sequence encoder
        dynamic_output, _ = self.dynamic_encoder(sequence_input)
        dynamic_feat = dynamic_output[:, -1, :]  # features of the last time step

        # fuse data
        fused = torch.cat([static_feat, dynamic_feat], dim=-1)

        output = self.predictor(fused)
        output = output.view(-1, self.output_horizon, self.output_dim)
        return output