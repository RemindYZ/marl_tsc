import torch
import torch.nn.functional as functional
from base import LocalStateEncoderBiLSTM, MemoryReader, MemoryWriter, Actor, Critic

class MemoryDDPG():
    def __init__(self, name, 
                 parameters, 
                 encoder_hyperparameters, 
                 critic_hyperparameters, 
                 actor_hyperparameters, 
                 device):
        self.name = name
        self.parameters = parameters
        self.encoder_hyperparameters = encoder_hyperparameters
        self.critic_hyperparameters = critic_hyperparameters
        self.actor_hyperparameters = actor_hyperparameters
        self.device = device
        self.memory_local = torch.tensor([0 for all in range(parameters["dim_memory"])], 
                                         dtype=torch.float64, device=device)
        self.encoder = LocalStateEncoderBiLSTM(encoder_hyperparameters["state_size"], 
                                               encoder_hyperparameters["hidden_size"],
                                               encoder_hyperparameters["num_layers"],
                                               encoder_hyperparameters["output_size"],
                                               encoder_hyperparameters["phase_size"],
                                               self.device)
        self.encoder_target = LocalStateEncoderBiLSTM(encoder_hyperparameters["state_size"], 
                                                      encoder_hyperparameters["hidden_size"],
                                                      encoder_hyperparameters["num_layers"],
                                                      encoder_hyperparameters["output_size"],
                                                      encoder_hyperparameters["phase_size"],
                                                      self.device)
        self.copy_all_parameters(self.encoder, self.encoder_target)

        self.reader = MemoryReader(critic_hyperparameters["state_size"], 
                                   parameters["dim_memory"],
                                   critic_hyperparameters["reader_h_size"],
                                   self.device)
        self.reader_target = MemoryReader(critic_hyperparameters["state_size"], 
                                          parameters["dim_memory"],
                                          critic_hyperparameters["reader_h_size"],
                                          self.device)
        self.copy_all_parameters(self.reader, self.reader_target)
        
        self.writer = MemoryWriter(critic_hyperparameters["state_size"],
                                   parameters["dim_memory"],
                                   self.device)
        self.writer_target = MemoryWriter(critic_hyperparameters["state_size"],
                                          parameters["dim_memory"],
                                          self.device)
        self.copy_all_parameters(self.writer, self.writer_target)

        self.critic = Critic(self.encoder, critic_hyperparameters["h_size"],
                             parameters["n_inter"],
                             self.device)
        self.critic_target = Critic(self.encoder_target, critic_hyperparameters["h_size"],
                                    parameters["n_inter"],
                                    self.device)
        self.copy_all_parameters(self.critic, self.critic_target)

        self.actor = Actor(self.encoder, self.reader, self.writer, 
                           actor_hyperparameters["h_size"],
                           actor_hyperparameters["action_size"],
                           parameters["n_neighbor"],
                           self.device)
        self.actor_target = Actor(self.encoder_target, self.reader_target, self.writer_target, 
                                  actor_hyperparameters["h_size"],
                                  actor_hyperparameters["action_size"],
                                  parameters["n_neighbor"],
                                  self.device)
        self.copy_all_parameters(self.actor, self.actor_target)

    def encoder_parameters_buffer(self):
        return self.encoder.state_dict()
    
    def copy_all_parameters(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())