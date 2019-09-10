import logging
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import time
from torch.optim import optimizer

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

class MemoryReader(nn.Module):
    def __init__(self, state_size, memory_size, h_size, device):
        super(MemoryRead, self).__init__()
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
