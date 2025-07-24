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
            nn.Linear(hidden_dim, output_horizon * output_dim)
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

class DynamicsGRUDataset(Dataset):
    def __init__(self, static_inputs, seq_inputs, targets):
        self.static_inputs = torch.from_numpy(static_inputs).float()
        self.seq_inputs = torch.from_numpy(seq_inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return self.seq_inputs.shape[0]

    def __getitem__(self, idx):
        return self.static_inputs, self.seq_inputs[idx], self.targets[idx]
    
class DynamicsGRU(nn.Module):
    def __init__(self, static_dim, seq_feature_dim, output_horizon, output_dim, hidden_dim=128, gru_layers=2):
        super().__init__()
        self.output_horizon = output_horizon
        self.output_dim = output_dim

        # static encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # dynamic sequence encoder（GRU）
        self.dynamic_encoder = nn.GRU(
            input_size=seq_feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_horizon * output_dim)
        )

    def forward(self, static_input, sequence_input):
        static_feat = self.static_encoder(static_input)  # [batch_size, hidden_dim]

        dynamic_output, _ = self.dynamic_encoder(sequence_input)
        dynamic_feat = dynamic_output[:, -1, :]  # last time step feature

        fused = torch.cat([static_feat, dynamic_feat], dim=-1)

        output = self.predictor(fused)
        output = output.view(-1, self.output_horizon, self.output_dim)
        return output
    
class DynamicsTransformerDataset(Dataset):
    def __init__(self, static_inputs, seq_inputs, targets):
        self.static_inputs = torch.from_numpy(static_inputs).float()
        self.seq_inputs = torch.from_numpy(seq_inputs).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return self.seq_inputs.shape[0]

    def __getitem__(self, idx):
        return self.static_inputs, self.seq_inputs[idx], self.targets[idx]
    
# class DynamicsTransformer(nn.Module):
#     def __init__(self, static_dim, seq_feature_dim, output_horizon, output_dim, hidden_dim=128, num_layers=2, nhead=8):
#         super().__init__()
#         self.output_horizon = output_horizon
#         self.output_dim = output_dim

#         # static encoder
#         self.static_encoder = nn.Sequential(
#             nn.Linear(static_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#         # input projection to match hidden_dim
#         self.input_projection = nn.Linear(seq_feature_dim, hidden_dim)

#         # dynamic sequence encoder（Transformer）
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim,
#             nhead=nhead,
#             dim_feedforward=hidden_dim * 4,
#             batch_first=True
#         )
#         # stack of N encoder layers
#         self.dynamic_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )

#         # predictor
#         self.predictor = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_horizon * output_dim)
#         )

#     def forward(self, static_input, sequence_input):
#         static_feat = self.static_encoder(static_input)  # [batch_size, hidden_dim]

#         # sequence projection
#         sequence_proj = self.input_projection(sequence_input)  # [batch_size, seq_len, hidden_dim]

#         # Transformer encoding
#         dynamic_encoded = self.dynamic_encoder(sequence_proj)  # [batch_size, seq_len, hidden_dim]

#         # average pooling over sequence
#         dynamic_feat = dynamic_encoded.mean(dim=1)  # [batch_size, hidden_dim]

#         # fusion
#         fused = torch.cat([static_feat, dynamic_feat], dim=-1)

#         # prediction
#         output = self.predictor(fused)
#         output = output.view(-1, self.output_horizon, self.output_dim)
#         return output

class DynamicsTransformer(nn.Module):
    def __init__(self, static_dim, seq_feature_dim, output_horizon, output_dim, hidden_dim=128, num_layers=2, nhead=8, max_seq_len=256):
        super().__init__()
        self.output_horizon = output_horizon
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # static encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # input projection
        self.input_projection = nn.Sequential(
            nn.Linear(seq_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))  # [1, T, D]
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.dynamic_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # prediction
        self.norm = nn.LayerNorm(2 * hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_horizon * output_dim)
        )

    def forward(self, static_input, sequence_input):
        N, T, _ = sequence_input.shape

        # static feature
        static_feat = self.static_encoder(static_input)  # [N, D]

        # project input
        seq_proj = self.input_projection(sequence_input)  # [N, T, D]

        # add positional encoding
        seq_proj = seq_proj + self.pos_embedding[:, :T, :]  # broadcast over batch

        # encode
        encoded_seq = self.dynamic_encoder(seq_proj)  # [N, T, D]
        # dynamic_feat = encoded_seq.mean(dim=1)  # [N, D]
        dynamic_feat = encoded_seq[:, -1, :]

        # fuse and predict
        # fused = torch.cat([static_feat, dynamic_feat], dim=-1)
        fused = self.norm(torch.cat([static_feat, dynamic_feat], dim=-1))
        output = self.predictor(fused)
        output = output.view(-1, self.output_horizon, self.output_dim)
        return output