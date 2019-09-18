import torch
import torch.nn.functional as functional
from torch import optim
from agents.base import LocalStateEncoderBiLSTM, MemoryReader, MemoryWriter, Actor, Critic

class MemoryDDPG():
    def __init__(self, name, 
                 encoder, encoder_target,
                 parameters, 
                 critic_hyperparameters, 
                 actor_hyperparameters, 
                 device):
        self.name = name
        self.parameters = parameters
        self.critic_hyperparameters = critic_hyperparameters
        self.actor_hyperparameters = actor_hyperparameters
        self.device = device
        self.memory_local = torch.tensor([0 for all in range(parameters["dim_memory"])], 
                                         dtype=torch.float, device=device)
        self.encoder = encoder
        self.encoder_target = encoder_target
        self.reader = MemoryReader(actor_hyperparameters["state_size"], 
                                   parameters["dim_memory"],
                                   actor_hyperparameters["reader_h_size"],
                                   self.device).to(self.device)
        self.reader_target = MemoryReader(actor_hyperparameters["state_size"], 
                                          parameters["dim_memory"],
                                          actor_hyperparameters["reader_h_size"],
                                          self.device).to(self.device)
        self.copy_all_parameters(self.reader, self.reader_target)
        
        self.writer = MemoryWriter(actor_hyperparameters["state_size"],
                                   parameters["dim_memory"],
                                   self.device).to(self.device)
        self.writer_target = MemoryWriter(actor_hyperparameters["state_size"],
                                          parameters["dim_memory"],
                                          self.device).to(self.device)
        self.copy_all_parameters(self.writer, self.writer_target)

        self.critic = Critic(self.encoder, critic_hyperparameters["h_size"],
                             parameters["n_inter"],
                             self.device).to(self.device)
        self.critic_target = Critic(self.encoder_target, critic_hyperparameters["h_size"],
                                    parameters["n_inter"],
                                    self.device).to(self.device)
        self.copy_all_parameters(self.critic, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_hyperparameters["learning_rate"])

        self.actor = Actor(self.encoder, self.reader, self.writer, 
                           actor_hyperparameters["h_size"],
                           actor_hyperparameters["action_size"],
                           parameters["n_neighbor"][name],
                           self.device).to(self.device)
        self.actor_target = Actor(self.encoder_target, self.reader_target, self.writer_target, 
                                  actor_hyperparameters["h_size"],
                                  actor_hyperparameters["action_size"],
                                  parameters["n_neighbor"][name],
                                  self.device).to(self.device)
        self.copy_all_parameters(self.actor, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_hyperparameters["learning_rate"])

    def encoder_parameters_buffer(self):
        return self.encoder.state_dict()
    
    def copy_all_parameters(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    def get_local_memory(self):
        return self.memory_local.clone().to(self.device)
    
    def update_local_memory(self, new_memory):
        self.memory_local = new_memory