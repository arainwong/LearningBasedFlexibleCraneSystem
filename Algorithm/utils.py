import os
import glob
from scipy.io import loadmat
import numpy as np
import torch
import yaml

def load_dataset(dataset_folder, command_type_set):
    # load all data from .mat files
    all_data = {}
    for command_type in command_type_set:
        current_dataset_folder = f'{dataset_folder}{command_type}'  # e.g. 'GeneratedDataset_Train/CommandType_sine'
        mat_files = glob.glob(f'{current_dataset_folder}/*.mat')
        # print(mat_files)
        for file_path in mat_files:
            key_name = os.path.basename(file_path).replace('.mat', '')
            # print(key_name)
            all_data[key_name] = loadmat(file_path)

    # combine all data into a dataset dictionary
    dataset = {}
    for key in all_data.keys():
        data = all_data[key]
        # remove keys name start with "__" in e.g. ['__header__', '__version__', '__globals__', 'Dataset_U_n']
        valid_keys = [k for k in data.keys() if not k.startswith('__')]
        last_key = valid_keys[-1]
        if last_key in dataset.keys():
            if last_key == 'Sys' or last_key == 'Body' or last_key == 'time_seq':
                continue
            dataset[last_key] = np.concatenate((dataset[last_key], data[last_key]), axis=0)
        else:
            dataset[last_key] = data[last_key]
        # print(f'{last_key}: {dataset[last_key].shape}')
    return dataset

def extract_sys_features(sys_raw):
    """
        [simulation_time, g, q1, q2, q3 (in deg.)]
        here only extract 'g'

        output dim: [1]
    """
    sys_data = sys_raw[0][0]
    features = [sys_data[1]] # g = -9.8
    
    return np.concatenate(features)

def extract_body_features(sys_raw):
    """
        [L, r, m, J, h]
        here only extract 'L', 'r', 'J'

        output dim: [48]
    """
    _, body_num = sys_raw.shape
    features = []
    for i in range(body_num-2):
        sys_data = sys_raw[0][i]
        # 扁平化数据
        features.append(np.atleast_1d(sys_data[0]).flatten()[0])  # L
        features.append(np.atleast_1d(sys_data[1]).flatten()[0])  # r
        if i != body_num-3:                                       # J, except for flexible beam
            features.append(np.atleast_1d(sys_data[2]).flatten()[0])
            features.append(sys_data[3][0, 0])
            features.append(sys_data[3][1, 1])
            features.append(sys_data[3][2, 2])
    
    sys_data = sys_raw[0][8]    # rope
    features.append(np.atleast_1d(sys_data[0]).flatten()[0])
    features.append(np.atleast_1d(sys_data[1]).flatten()[0])
    features.append(np.atleast_1d(sys_data[2]).flatten()[0])
    features.append(sys_data[3][0, 0])
    features.append(sys_data[3][1, 1])
    features.append(sys_data[3][2, 2])

    sys_data = sys_raw[0][9]    # payload
    features.append(np.atleast_1d(sys_data[1]).flatten()[0])
    features.append(np.atleast_1d(sys_data[2]).flatten()[0])
    features.append(sys_data[3][0, 0])
    features.append(sys_data[3][1, 1])
    features.append(sys_data[3][2, 2])

    return np.array(features).flatten()

def prepare_MLP_dataset(dataset, input_horizon, output_horizon):
    """
        features: [current_t, Command_n[0:3], U_n[0:9], U_n[15:18], V_n..., A_n..., P_n[0:6], dP_n[0:6], Q_n[0:6]]
        U_n[0:9] -> q1, ..., q9
        U_n[15:18] -> EE_x, EE_y, EE_z
        P_n[0:6] -> p1_in, p1_out, p2_in, p2_out, p3_in, p3_out

        inputs: [N * samples, single_feature_dim * input_horizon] -> 58[t, command, u, v, a, p, dp, q] * input_horizon + 1[sys.g] + 48[body]
        targets: [N * samples, single_target_dim * output_horizon]
    """

    target_key = 'Dataset_U_n'
    target_set = dataset[target_key]  # [N, dim_U, time_step]
    N, dim, time_step = target_set.shape
    
    input_list = []
    target_list = []
    
    for t in range(time_step - input_horizon - output_horizon + 1):
        current_inputs = []

        time_seq = dataset['time_seq'][:, t:t+input_horizon]  # shape: [1, input_horizon]
        time_seq = np.tile(time_seq, (N, 1))         # shape: [N, input_horizon]
        current_inputs.append(time_seq)

        sys_features = extract_sys_features(dataset['Sys'])   # g = -9.8
        sys_features = np.tile(sys_features, (N, 1))
        current_inputs.append(sys_features)

        body_features = extract_body_features(dataset['Body']) # [body[1:8] -> L, r; body[9] -> L, r, m; body[10] -> r, m]
        body_features = np.tile(body_features, (N, 1))
        current_inputs.append(body_features)

        Command_xv1_to_xv3 = dataset['Dataset_Command_n'][:, 0:3, t:t+input_horizon]

        U_q1_to_q9 = dataset['Dataset_U_n'][:, 0:9, t:t+input_horizon]
        U_EEx_to_EEz = dataset['Dataset_U_n'][:, 15:18, t:t+input_horizon]
        V_q1_to_q9 = dataset['Dataset_V_n'][:, 0:9, t:t+input_horizon]
        V_EEx_to_EEz = dataset['Dataset_V_n'][:, 15:18, t:t+input_horizon]
        A_q1_to_q9 = dataset['Dataset_A_n'][:, 0:9, t:t+input_horizon]
        A_EEx_to_EEz = dataset['Dataset_A_n'][:, 15:18, t:t+input_horizon]

        P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon]
        dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon]
        Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon]
    
        current_inputs.extend([Command_xv1_to_xv3, 
                                U_q1_to_q9, U_EEx_to_EEz, V_q1_to_q9, V_EEx_to_EEz, A_q1_to_q9, A_EEx_to_EEz, 
                                P_1_to_3, dP_1_to_3, Q_1_to_3])
        current_inputs_flattened = [var.reshape(var.shape[0], -1) for var in current_inputs]
        input_t = np.concatenate(current_inputs_flattened, axis=1)  # [N, features * input_horizon]
        
        # EEx to EEz
        target_t = target_set[:, 15:18, t+input_horizon:t+input_horizon+output_horizon]
        target_t = target_t.reshape(N, -1)  # [N, target * output_horizon]
        
        input_list.append(input_t)
        target_list.append(target_t)
    
    # stack all time steps
    inputs  = np.concatenate(input_list, axis=0)   # [N*(time_step-input_output_horizon), features * input_horizon]
    targets = np.concatenate(target_list, axis=0)  # [N*(time_step-input_output_horizon), target * output_horizon]
    
    return inputs, targets

def prepare_static_dataset(dataset):
    """
        sys_features dim: 1
        body_features dim: 55 [L, r, m, J]
    """
    sys_features = extract_sys_features(dataset['Sys']).flatten()   # g = -9.8
    body_features = extract_body_features(dataset['Body']).flatten() # [body[1:8] -> L, r; body[9] -> L, r, m; body[10] -> r, m]
    static_features = np.concatenate([sys_features, body_features])
    return static_features

def prepare_dynamic_sequence_dataset(dataset, input_horizon, output_horizon, data_config, target_config):
    """
        input: [samples, input_horizon, feature_dim] -> feature_dim = 58 -> [t, command, u, v, a, p, dp, q]
        output: [samples, output_horizon, target_dim]
    """

    N, dim, time_step = dataset['Dataset_U_n'].shape    # [N, dim_U, time_step]

    input_list = []
    target_list = []

    for t in range(time_step - input_horizon - output_horizon + 1):
        # t, xv1, xv2, xv3 -> dim = 4
        time_seq = dataset['time_seq'][:, t:t+input_horizon]         # [1, input_horizon]
        time_seq = np.tile(time_seq, (N, 1))                         # [N, input_horizon]
        time_seq = time_seq[..., np.newaxis]                         # [N, input_horizon, 1]
        Command_xv1_to_xv3 = dataset['Dataset_Command_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)

        # [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in U, V, A; P, Q -> dim = 42
        if data_config == 'data_config_0':
            U_q1_to_q3 = dataset['Dataset_U_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)       # 3
            U_d5 = dataset['Dataset_U_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_d7 = dataset['Dataset_U_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_phi1_to_phi2 = dataset['Dataset_U_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)   # 2
            U_EEx_to_EEz = dataset['Dataset_U_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)   # 3

            V_q1_to_q3 = dataset['Dataset_V_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)
            V_d5 = dataset['Dataset_V_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)
            V_d7 = dataset['Dataset_V_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)
            V_phi1_to_phi2 = dataset['Dataset_V_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            V_EEx_to_EEz = dataset['Dataset_V_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)
            
            A_q1_to_q3 = dataset['Dataset_A_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)
            A_d5 = dataset['Dataset_A_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)
            A_d7 = dataset['Dataset_A_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)
            A_phi1_to_phi2 = dataset['Dataset_A_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            A_EEx_to_EEz = dataset['Dataset_A_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)

            P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6
            # dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)
            Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6

            # concatenate features to [N, input_horizon, feature_dim]
            input_t = np.concatenate([
                time_seq,
                Command_xv1_to_xv3,
                U_q1_to_q3, U_d5, U_d7, U_phi1_to_phi2, U_EEx_to_EEz,
                V_q1_to_q3, V_d5, V_d7, V_phi1_to_phi2, V_EEx_to_EEz,
                A_q1_to_q3, A_d5, A_d7, A_phi1_to_phi2, A_EEx_to_EEz,
                P_1_to_3, Q_1_to_3
            ], axis=2)
            input_information = f't; xv1, xv2, xv3; [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in U, V, A; P, Q.'

        # [theta1, theta2, theta3, d5, d7] in U; [phi1, phi2, EEx, EEy, EEz] in A; P, Q -> dim = 22
        elif data_config == 'data_config_1':
            U_q1_to_q3 = dataset['Dataset_U_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)       # 3
            U_d5 = dataset['Dataset_U_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_d7 = dataset['Dataset_U_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)             # 1

            A_phi1_to_phi2 = dataset['Dataset_A_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            A_EEx_to_EEz = dataset['Dataset_A_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)

            P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6
            # dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)
            Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6

            input_t = np.concatenate([
                time_seq,
                Command_xv1_to_xv3,
                U_q1_to_q3, U_d5, U_d7,
                A_phi1_to_phi2, A_EEx_to_EEz,
                P_1_to_3, Q_1_to_3
            ], axis=2)
            input_information = f't; xv1, xv2, xv3; [theta1, theta2, theta3, d5, d7] in U; [phi1, phi2, EEx, EEy, EEz] in A; P, Q.'

        # [theta1, theta2, theta3, d5, d7] in U; [phi1, phi2, EEx, EEy, EEz] in V; P, Q -> dim = 22
        elif data_config == 'data_config_2':
            U_q1_to_q3 = dataset['Dataset_U_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)       # 3
            U_d5 = dataset['Dataset_U_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_d7 = dataset['Dataset_U_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)             # 1

            V_phi1_to_phi2 = dataset['Dataset_V_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            V_EEx_to_EEz = dataset['Dataset_V_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)

            P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6
            # dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)
            Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6

            input_t = np.concatenate([
                time_seq,
                Command_xv1_to_xv3,
                U_q1_to_q3, U_d5, U_d7,
                V_phi1_to_phi2, V_EEx_to_EEz,
                P_1_to_3, Q_1_to_3
            ], axis=2)
            input_information = f't; xv1, xv2, xv3; [theta1, theta2, theta3, d5, d7] in U; [phi1, phi2, EEx, EEy, EEz] in V; P, Q.'

        # [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in U; P, Q -> dim = 22
        elif data_config == 'data_config_3':
            U_q1_to_q3 = dataset['Dataset_U_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)       # 3
            U_d5 = dataset['Dataset_U_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_d7 = dataset['Dataset_U_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)             # 1
            U_phi1_to_phi2 = dataset['Dataset_U_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            U_EEx_to_EEz = dataset['Dataset_U_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)

            P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6
            # dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)
            Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6

            input_t = np.concatenate([
                time_seq,
                Command_xv1_to_xv3,
                U_q1_to_q3, U_d5, U_d7,
                U_phi1_to_phi2, U_EEx_to_EEz,
                P_1_to_3, Q_1_to_3
            ], axis=2)
            input_information = f't; xv1, xv2, xv3; [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in U; P, Q.'

        # [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in V; P, Q -> dim = 22
        elif data_config == 'data_config_4':
            V_q1_to_q3 = dataset['Dataset_V_n'][:, 0:3, t:t+input_horizon].transpose(0, 2, 1)       # 3
            V_d5 = dataset['Dataset_V_n'][:, 4:5, t:t+input_horizon].transpose(0, 2, 1)             # 1
            V_d7 = dataset['Dataset_V_n'][:, 6:7, t:t+input_horizon].transpose(0, 2, 1)             # 1
            V_phi1_to_phi2 = dataset['Dataset_V_n'][:, 7:9, t:t+input_horizon].transpose(0, 2, 1)
            V_EEx_to_EEz = dataset['Dataset_V_n'][:, 15:18, t:t+input_horizon].transpose(0, 2, 1)

            P_1_to_3 = dataset['Dataset_P_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6
            # dP_1_to_3 = dataset['Dataset_dP_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)
            Q_1_to_3 = dataset['Dataset_Q_n'][:, 0:6, t:t+input_horizon].transpose(0, 2, 1)         # 6

            input_t = np.concatenate([
                time_seq,
                Command_xv1_to_xv3,
                V_q1_to_q3, V_d5, V_d7,
                V_phi1_to_phi2, V_EEx_to_EEz,
                P_1_to_3, Q_1_to_3
            ], axis=2)
            input_information = f't; xv1, xv2, xv3; [theta1, theta2, theta3, d5, d7, phi1, phi2, EEx, EEy, EEz] in V; P, Q.'

        # phi1, phi2, EE_x, EE_y, EE_z
        if target_config == 'target_U':
            target_phi1_to_phi2 = dataset['Dataset_U_n'][:, 7:9, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_EE = dataset['Dataset_U_n'][:, 15:18, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_information = f'[phi1, phi2, EEx, EEy, EEz] in U.'
        
        elif target_config == 'target_V':
            target_phi1_to_phi2 = dataset['Dataset_V_n'][:, 7:9, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_EE = dataset['Dataset_V_n'][:, 15:18, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_information = f'[phi1, phi2, EEx, EEy, EEz] in V.'

        elif target_config == 'target_A':
            target_phi1_to_phi2 = dataset['Dataset_A_n'][:, 7:9, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_EE = dataset['Dataset_A_n'][:, 15:18, t+input_horizon:t+input_horizon+output_horizon].transpose(0, 2, 1)
            target_information = f'[phi1, phi2, EEx, EEy, EEz] in A.'

        target_t = np.concatenate([
            target_phi1_to_phi2,
            target_EE
        ], axis=2)  # [N, output_horizon, features]

        input_list.append(input_t)
        target_list.append(target_t)

    # stack all samples
    inputs = np.concatenate(input_list, axis=0)    # [samples, input_horizon, features_dim]
    targets = np.concatenate(target_list, axis=0)  # [samples, output_horizon, features_dim]
    print(f'Input: {input_information}')
    print(f'Target: {target_information}')

    return inputs, targets

class Normalizer:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.mean = None
        self.std = None

    def numpy_to_tensor(self):
        self.mean = torch.from_numpy(self.mean).float()  # ensure float32
        self.std = torch.from_numpy(self.std).float()

    def tensor_to_numpy(self):
        self.mean = self.mean.cpu().numpy()
        self.std = self.std.cpu().numpy()

    def fit(self, data):
        """
        Handle data shape [..., D], e.g., [N, T, D] or [N, D]
        """
        if isinstance(data, np.ndarray):
            dims = tuple(i for i in range(data.ndim - 1))
            self.mean = np.mean(data, axis=dims, keepdims=True).astype(np.float32)  # ensure float32
            self.std = np.std(data, axis=dims, keepdims=True).astype(np.float32)
        elif isinstance(data, torch.Tensor):
            dims = tuple(i for i in range(data.ndim - 1))
            self.mean = torch.mean(data, dim=dims, keepdim=True).float()  # ensure float32
            self.std = torch.std(data, dim=dims, keepdim=True).float()
        else:
            raise TypeError("Unsupported data type for fitting.")

    def normalize(self, data):
        if isinstance(data, np.ndarray):
            return (data - self.mean) / (self.std + self.eps)
        elif isinstance(data, torch.Tensor):
            return (data - self.mean.to(data.device).float()) / (self.std.to(data.device).float() + self.eps)
        else:
            raise RuntimeError("Normalizer not fitted yet.")

    def denormalize(self, data):
        """
            To original data
        """
        if isinstance(data, np.ndarray):
            return data * (self.std + self.eps) + self.mean
        elif isinstance(data, torch.Tensor):
            return data * (self.std.to(data.device).float() + self.eps) + self.mean.to(data.device).float()
        else:
            raise RuntimeError("Normalizer not fitted yet.")
        
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_a_sample(dataset, index):
    print(dataset[index])

def show_dataset_shape(dataset):
    for key in dataset.keys():
        print(f'{key}: {dataset[key].shape}')

def INDEX_TEST(max_value, start, end):
    data = [i for i in range(1, max_value)]
    print(data, len(data))
    print(data[start:end])
