import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

class LocalStateEncoderBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, phase_size, device):
        super(LocalStateEncoderBiLSTM, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.phase_size = phase_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2 + phase_size, output_size)
    
    def forward(self, state, phase):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, state.size(0), self.hidden_size).to(self.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, state.size(0), self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(state, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(torch.cat((out[:, -1, :], phase), dim=1))
        return out

class LocalStateEncoderCNN(nn.Module):
    def __init__(self, output_size, phase_size, device):
        super(LocalStateEncoderCNN, self).__init__()
        self.device = device
        self.output_size = output_size
        self.phase_size = phase_size
        self.device = device
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(1, 4, 3),
            nn.Conv2d(4, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(8*16 + phase_size, output_size)
    
    def forward(self, state, phase):
        out = self.cnn_layer1(state.reshape(-1, 1, 20, 6))
        out = out.reshape(out.size(0), -1)
        out = self.fc(torch.cat((out, phase), dim=1))
        return out

class MemoryReader(nn.Module):
    def __init__(self, state_size, memory_size, h_size, device):
        super(MemoryReader, self).__init__()
        self.device = device
        self.state_size = state_size
        self.memory_size = memory_size
        self.h_size = h_size
        self.fc_h = nn.Linear(state_size, h_size)
        self.fc_k = nn.Linear(state_size + h_size + memory_size, memory_size)
    
    def forward(self, state, memory):
        h = self.fc_h(state)
        k = self.fc_k(torch.cat((state, h, memory), dim=1)).sigmoid()
        out = memory * k
        return out

class MemoryWriter(nn.Module):
    def __init__(self, state_size, memory_size, device):
        super(MemoryWriter, self).__init__()
        self.device = device
        self.state_size = state_size
        self.memory_size = memory_size
        self.fc_r = nn.Linear(state_size + memory_size, memory_size)
        self.fc_z = nn.Linear(state_size + memory_size, memory_size)
        self.fc_c = nn.Linear(state_size + memory_size, memory_size)
    
    def forward(self, state, memory):
        r = self.fc_r(torch.cat((state, memory), dim=1)).sigmoid()
        z = self.fc_z(torch.cat((state, memory), dim=1)).sigmoid()
        c = self.fc_c(torch.cat((state, r * memory), dim=1)).tanh()
        out = (1 - z) * memory + z * c
        return out

class Actor(nn.Module):
    def __init__(self, encoder_hyperparameters, dim_memory, reader_h_size, 
                 h_size, action_size, n_neighbor, device):
        super(Actor, self).__init__()
        self.device = device
        # self.encoder = LocalStateEncoderBiLSTM(encoder_hyperparameters["state_size"], 
        #                                        encoder_hyperparameters["hidden_size"],
        #                                        encoder_hyperparameters["num_layers"],
        #                                        encoder_hyperparameters["output_size"],
        #                                        encoder_hyperparameters["phase_size"],
        #                                        device)
        self.encoder = LocalStateEncoderCNN(encoder_hyperparameters["output_size"],
                                            encoder_hyperparameters["phase_size"],
                                            device)
        self.reader = MemoryReader(encoder_hyperparameters["output_size"], 
                                   dim_memory,
                                   reader_h_size,
                                   device)
        self.writer = MemoryWriter(encoder_hyperparameters["output_size"],
                                   dim_memory,
                                   device)
        self.h_size = h_size
        self.action_size = action_size
        self.n_neighbor = n_neighbor
        inp_size = [self.encoder.output_size + self.reader.memory_size * (2 + n_neighbor)] + h_size[:-1]
        self.fc = [nn.Linear(inp, oup).to(device) for inp, oup in zip(inp_size, h_size)]
        self.out_fc = nn.Linear(h_size[-1], action_size)
    
    def forward(self, state, phase, local_memory, neighbor_memory):
        e = self.encoder(state, phase)
        r_l = self.reader(e, local_memory)
        m_l = self.writer(e, local_memory)
        r_N = [self.reader(e, n_m) for n_m in neighbor_memory]
        out = torch.cat([e, r_l, m_l] + r_N, dim=1)
        for fc in self.fc:
            out = fc(out).relu()
        if self.action_size == 1:
            out = self.out_fc(out).sigmoid()
        else:
            out = self.out_fc(out).softmax()
        return out, m_l

class Critic(nn.Module):
    def __init__(self, encoder_hyperparameters, h_size, n_inter, device):
        super(Critic, self).__init__()
        self.device = device
        # self.encoder = LocalStateEncoderBiLSTM(encoder_hyperparameters["state_size"], 
        #                                        encoder_hyperparameters["hidden_size"],
        #                                        encoder_hyperparameters["num_layers"],
        #                                        encoder_hyperparameters["output_size"],
        #                                        encoder_hyperparameters["phase_size"],
        #                                        device)
        self.encoder = LocalStateEncoderCNN(encoder_hyperparameters["output_size"],
                                            encoder_hyperparameters["phase_size"],
                                            device)
        self.h_size = h_size
        self.n_inter = n_inter
        self.device = device
        inp_size = [(self.encoder.output_size + 1) * n_inter] + h_size[:-1]
        self.fc = [nn.Linear(inp, oup).to(device) for inp, oup in zip(inp_size, h_size)]
        self.out_fc = nn.Linear(h_size[-1], 1)
    
    def forward(self, state, phase, action):
        e =[self.encoder(state[:,:,:,ind], phase[:,:,ind]) for ind in range(state.shape[-1])]
        inp = [torch.cat((e[ind], action[:,ind].reshape(-1,1)), dim=1) for ind in range(state.shape[-1])]
        out = torch.cat(inp, dim=1)
        for fc in self.fc:
            out = fc(out).relu()
        out = self.out_fc(out)
        return out