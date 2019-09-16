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

        self.state_height = int(self.env.ild_length/self.env.ver_length)
        # self.state_width = 6
        self.phase_size = 4

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
                    states, phases, actions, rewards, next_states, next_phases, local_memorys, dones = self.replaybuffer.sample()
                    self.critic_learn(states, phases, actions, rewards, next_states, next_phases, local_memorys, dones)
                    self.actor_learn(states, phases, local_memorys)
            states, phases, actions, rewards, n_states, n_phases, memory = self._dict_to_numpy(self.obs, self.actions, self.rewards, self.next_obs, self.memory)
            self.replaybuffer.add_experience(states, phases, actions, rewards, n_states, n_phases, memory, self.dones)
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
    
    def _dict_to_numpy(self, obs, actions, rewards, next_obs, memory):
        states, phases = obs
        n_states, n_phases = next_obs
        state_numpy = np.stack([states[a_n] for a_n in self.actor_names], axis=2).reshape(-1)
        phase_numpy = np.stack([phases[a_n] for a_n in self.actor_names], axis=1).reshape(-1)
        action_numpy = np.stack([actions[a_n] for a_n in self.actor_names], axis=0).reshape(-1)
        reward_numpy = np.stack([rewards[a_n] for a_n in self.actor_names], axis=0).reshape(-1)
        n_state_numpy = np.stack([n_states[a_n] for a_n in self.actor_names], axis=2).reshape(-1)
        n_phase_numpy = np.stack([n_phases[a_n] for a_n in self.actor_names], axis=1).reshape(-1)
        memory_numpy = np.stack([memory[a_n] for a_n in self.actor_names], axis=1).reshape(-1)
        return state_numpy, phase_numpy, action_numpy, reward_numpy, n_state_numpy, n_phase_numpy, memory_numpy
    
    def critic_learn(self, states, phases, actions, rewards, next_states, next_phases, local_memorys, dones):
        loss = self.compute_loss(states, phases, actions, rewards, next_states, next_phases, local_memorys, dones)
        self._optim_critic(loss)
        tau = self.parameters["critic_tau"]
        for actor in self.actors:
            for f_model, t_model in zip(actor.critic.parameters(), actor.critic_target.parameters()):
                t_model.data.copy_(tau*f_model.data+(1-tau)*t_model.data)

    def actor_learn(self, states, phases, local_memorys):
        if self.dones:
            #updata learning rate
            pass
        state = states.reshape(-1, self.state_height, self.encoder_hyperparameters["state_size"], self.parameters["n_inter"])
        phase = phases.reshape(-1, self.phase_size, self.parameters["n_inter"])
        local_memory = local_memorys.reshape(-1, self.parameters["dim_memory"], self.parameters["n_inter"])
        neighbor_map = [[self.actor_names.index(neigh) for neigh in self.env.neighbor_map[ac_name]] 
                        for ac_in, ac_name in enumerate(self.actor_names)]
        neighbor_memory = [local_memory[:,:,ind] for ind in neighbor_map]
        actions_pred = torch.cat([self.actors[ac_in].actor(state[:,:,:,ac_in], phase[:,:,ac_in], local_memory[:,:,ac_in], neighbor_memory[ac_in]) 
                            for ac_in, _ in enumerate(self.actor_names)], dim=1).reshape(-1, self.parameters["n_inter"])
        action_loss = [-self.actors[i].critic(state, phase, actions_pred).mean() for i in range(len(self.parameters["n_inter"]))]
        self._optim_actor(action_loss)
        tau = self.parameters["actor_tau"]
        for actor in self.actors:
            for f_model, t_model in zip(actor.actor.parameters(), actor.actor_target.parameters()):
                t_model.data.copy_(tau*f_model.data+(1-tau)*t_model.data)

    def compute_loss(self, states, phases, actions, rewards, next_states, next_phases, local_memorys, dones):
        state = states.reshape(-1, self.state_height, self.encoder_hyperparameters["state_size"], self.parameters["n_inter"])
        phase = phases.reshape(-1, self.phase_size, self.parameters["n_inter"])
        action = actions.reshape(-1, self.parameters["n_inter"])
        reward = rewards.reshape(-1, self.parameters["n_inter"])
        next_state = next_states.reshape(-1, self.state_height, self.encoder_hyperparameters["state_size"], self.parameters["n_inter"])
        next_phase = next_phases.reshape(-1, self.phase_size, self.parameters["n_inter"])
        local_memory = local_memorys.reshape(-1, self.parameters["dim_memory"], self.parameters["n_inter"])
        neighbor_map = [[self.actor_names.index(neigh) for neigh in self.env.neighbor_map[ac_name]] 
                        for ac_in, ac_name in enumerate(self.actor_names)]
        neighbor_memory = [local_memory[:,:,ind] for ind in neighbor_map]
        done = dones.reshape(-1,1)
        with torch.no_grad():
            actions_next = torch.cat([self.actors[ac_in].actor_target(next_state[:,:,:,ac_in], next_phase[:,:,ac_in], local_memory[:,:,ac_in], neighbor_memory[ac_in]) 
                            for ac_in, _ in enumerate(self.actor_names)], dim=1).reshape(-1, self.parameters["n_inter"])
            critic_targets_next = torch.cat([self.actors[ac_in].critic_target(next_state, next_phase, actions_next) 
                                   for ac_in, _ in enumerate(self.actor_names)], dim=1).reshape(-1, self.parameters["n_inter"])
        critic_targets = reward + self.parameters["gamma"] * critic_targets_next * (1.0 - done)
        crititc_expected = torch.cat([self.actors[ac_in].critic(state, phase, action) 
                                      for ac_in, _ in enumerate(self.actor_names)], dim=1).reshape(-1, self.parameters["n_inter"])
        loss = [functional.mse_loss(crititc_expected[:,i], critic_targets[:,i]) for i in range(critic_targets.shape[-1])]
        return loss
    
    def _optim_critic(self, loss, clipping_norm=None):
        for actor_ind, actor in enumerate(self.actors):
            actor.critic_optimizer.zero_grad()
            loss[actor_ind].backward()
            if clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor.critic.parameters(), clipping_norm)
            actor.critic_optimizer.step()
    
    def _optim_actor(self, loss, clipping_norm=None):
        for actor_ind, actor in enumerate(self.actors):
            actor.actor_optimizer.zero_grad()
            loss[actor_ind].backward()
            if clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), clipping_norm)
            actor.actor_optimizer.step()




        