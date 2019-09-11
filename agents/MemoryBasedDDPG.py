import torch
import torch.nn.functional as functional
from torch import optim
from utilities.ReplayBuffer import Replay_Buffer
from base import LocalStateEncoderBiLSTM, MemoryReader, MemoryWriter

class MemoryDDPG():
    def __init__(self, encoder_parameters, critic_hyperparameters, actor_hyperparameters, device):
        self.encoder_parameters = encoder_parameters
        self.critic_hyperparameters = critic_hyperparameters
        self.actor_hyperparameters = actor_hyperparameters
        self.device = device
        self.encoder = LocalStateEncoderBiLSTM(encoder_parameters["state_size"], 
                                               encoder_parameters["hidden_size"],
                                               encoder_parameters["num_layers"],
                                               encoder_parameters["output_size"],
                                               encoder_parameters["phase_size"],
                                               self.device)
    
    def encoder_parameters_buffer(self):
        return self.encoder.state_dict()
    
    def copy_all_parameters(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    def copy_encoder_parameters(self, encoder_model):
        for to_model, from_model in zip(self.encoder.parameters(), encoder_model.parameters()):
            to_model.data.copy_(from_model.data.clone())