import torch
import torch.nn.functional as functional
from torch.distributions import Categorical
from torch import optim
import random
import numpy as np
from utilities.ReplayBuffer import Replay_Buffer
from utilities.logger import Logger
from MemoryBasedDDPG import MemoryDDPG
from base import LocalStateEncoderBiLSTM

class MA_MDDPG():
    def __init__(self, env, parameters,
                 encoder_hyperparameters, 
                 critic_hyperparameters, 
                 actor_hyperparameters):
        self.env = env
        self.neighbor_map = env.neighbor_map
        self.parameters = parameters
        self.encoder_hyperparameters = encoder_hyperparameters
        self.critic_hyperparameters = critic_hyperparameters
        self.actor_hyperparameters = actor_hyperparameters
        self.device = "cuda:0"
        self.logger = Logger(self.parameters["log_dir"])
        self.set_random_seeds(parameters["random_seed"])
        self.epsilon_exploration = parameters["epsilon_exploration"]
        self.replaybuffer = Replay_Buffer(parameters["buffer_size"], parameters["batch_size"], parameters["random_seed"])
        self.encoder =  LocalStateEncoderBiLSTM(encoder_hyperparameters["state_size"], 
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
        self.copy_encoder_parameters(self.encoder, self.encoder_target)

        self.actor_names = env.nodes_name
        self.actors = [MemoryDDPG(actor, self.encoder, self.encoder_target, parameters, critic_hyperparameters,
                       actor_hyperparameters, self.device) for actor in self.actor_names]
        self.memory = {actor.name:actor.get_local_memory() for actor in self.actors}
        self.global_step_number = 0
        self.episode_number = 0

    
    def set_random_seeds(self, random_seed):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)
    
    def update_memory(self):
        self.memory = [actor.get_local_memory() for actor in self.actors]
    
    def get_neighbor_memory(self, name):
        neighbor = self.neighbor_map[name]
        return [self.memory[self.actors.index(n_b)].clone() for n_b in neighbor]
    
    def get_local_memory(self, name):
        return self.memory[self.actors.index(name)].clone()
    
    def copy_encoder_parameters(self, from_model, to_model):
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
    
    def step(self):
        self.obs = self.env.reset()
        while True:
            self.actions = self._pick_action(self.obs)
            self.next_obs, self.rewards, self.dones, _ = self.env.step(self.actions)
            if len(self.replaybuffer) >= self.parameters["batch_size"]:
                for _ in range(self.parameters["learning_step_per_session"]):
                    states, actions, rewards, next_states, local_memorys, dones = self.replaybuffer.sample()
                    self.critic_learn()
                    self.actor_learn()
            self.replaybuffer.add_experience(self.obs, self.actions, self.rewards, self.next_obs, self.memory, self.dones)
            self.obs = self.next_obs
            self.global_step_number += 1
            if self.dones == True: break
        self.episode_number += 1
                
    
    def _pick_action(self, obs=None):
        # if obs is None: obs = self.env.reset()
        position, phase = obs
        actions = {}
        all_new_memory = []
        for actor_ind, actor_name in enumerate(self.actor_names):
            po, ph = position[actor_name], phase[actor_name]
            n_memory = self.get_neighbor_memory(actor_name)
            l_memory = self.get_local_memory(actor_name)
            self.actors[actor_ind].actor.eval()
            with torch.no_grad():
                action_output, new_memory = self.actors[actor_ind].actor(po, ph, l_memory, n_memory)
                action_distribution = Categorical(action_output)
                action = action_distribution.sample().cpu().numpy()
            self.actors[actor_ind].actor.train()
            if random.random() <= self.epsilon_exploration:
                action = random.randint(0, 1)
            else:
                action = action[0]
            actions[actor_name] = action 
            all_new_memory.append(new_memory)
        for actor_ind, actor in enumerate(self.actors):
            actor.update_memory(all_new_memory[actor_ind])
        self.memory = {actor.name:actor.get_local_memory() for actor in self.actors}
        return actions
    
    def critic_learn(self):
        pass

    def actor_learn(self):
        pass



        