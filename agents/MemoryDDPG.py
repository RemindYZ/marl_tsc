import torch
import torch.nn.functional as functional
from torch import optim
from agents.base import LocalStateEncoderBiLSTM, LocalStateEncoderCNN, MemoryReader, MemoryWriter, Actor, Critic

class MemoryDDPG():
    def __init__(self, name, 
                 parameters, 
                 encoder_hyperparameters,
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

        self.critic = Critic(encoder_hyperparameters, critic_hyperparameters["h_size"],
                             parameters["n_inter"],
                             self.device).to(self.device)
        self.critic_target = Critic(encoder_hyperparameters, critic_hyperparameters["h_size"],
                                    parameters["n_inter"],
                                    self.device).to(self.device)
        self.copy_all_parameters(self.critic, self.critic_target)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_hyperparameters["learning_rate"])

        self.actor = Actor(encoder_hyperparameters, parameters["dim_memory"],
                           actor_hyperparameters["reader_h_size"],
                           actor_hyperparameters["h_size"],
                           actor_hyperparameters["action_size"],
                           parameters["n_neighbor"][name],
                           self.device).to(self.device)
        self.actor_target = Actor(encoder_hyperparameters, parameters["dim_memory"],
                                  actor_hyperparameters["reader_h_size"],
                                  actor_hyperparameters["h_size"],
                                  actor_hyperparameters["action_size"],
                                  parameters["n_neighbor"][name],
                                  self.device).to(self.device)
        self.copy_all_parameters(self.actor, self.actor_target)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_hyperparameters["learning_rate"])

    def critic_encoder_parameters(self):
        return self.critic.encoder.state_dict()
    
    def actor_encoder_parameters(self):
        return self.actor.encoder.state_dict()
    
    def load_encoder_parameters(self, encoder=None, encoder_target=None):
        if not encoder is None:
            self.copy_all_parameters(encoder, self.critic.encoder)
            self.copy_all_parameters(encoder, self.actor.encoder)
        if not encoder_target is None:
            self.copy_all_parameters(encoder_target, self.critic_target.encoder)
            self.copy_all_parameters(encoder_target, self.actor_target.encoder)
        return self
    
    def copy_all_parameters(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    def get_local_memory(self):
        return self.memory_local.clone().to(self.device)
    
    def update_local_memory(self, new_memory):
        self.memory_local = new_memory