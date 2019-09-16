from collections import namedtuple, deque
import torch
import random
import numpy as np

class Replay_Buffer():
    def __init__(self, buffer_size, batch_size, seed = None):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "phase", "action", "reward",
                                     "next_state", "next_phase", "local_memory", "done"])
        if seed:
            self.seed = random.seed(seed)
        else:
            self.seed = 42
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.memory)
    
    def add_experience(self, states, phases, actions, rewards, next_states, next_phases, local_memorys, dones):
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(state, phase, action, reward, next_state, next_phase, local_memory, done)
                           for state, phase, action, reward, next_state, next_phase, local_memory, done in
                           zip(states, phases, actions, rewards, next_states, next_phases, local_memorys, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, phases, actions, rewards, next_states, next_phases, local_memorys, dones)
            self.memory.append(experience)
    
    def sample(self, num_experiences=None, separate_out_data_types=True):
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, phases, actions, rewards, next_states, next_phases, local_memorys, dones = self.separate_out_data_types(experiences)
            return states, phases, actions, rewards, next_states, next_phases, local_memorys, dones
        else:
            return experiences

    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        phases = torch.from_numpy(np.vstack([e.phase for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_phases = torch.from_numpy(np.vstack([e.next_phase for e in experiences if e is not None])).float().to(self.device)
        local_memorys = torch.from_numpy(np.vstack([e.local_memory for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
        return states, phases, actions, rewards, next_states, next_phases, local_memorys, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)