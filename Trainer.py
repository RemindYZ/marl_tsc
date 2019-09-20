import torch
import torch.nn.functional as functional
import random
import numpy as np
from agents.MA_MDDPG import MA_MDDPG
from env.env import TrafficEnv

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parameters = {
    "name": "CNN_MA2C", 
    "random_seed": 42,
    "log_dir": "./experiments/logs/",
    "model_dir": "./experiments/model/",
    "epsilon_exploration": 0.3,
    "buffer_size": 10000,
    "batch_size": 256,
    "n_inter": 9,
    "learning_step_per_session": 1,
    "critic_tau": 0.995, 
    "actor_tau": 0.995,
    "dim_memory": 16,
    "gamma": 0.99,
    "n_neighbor" : {
        "I0": 2, "I1": 3, "I2": 2, 
        "I3": 3, "I4": 4, "I5": 3,
        "I6": 2, "I7": 3, "I8": 2
    }
}
encoder_hyperparameters = {
    "state_size": 6, 
    "hidden_size": 128,
    "num_layers": 1,
    "output_size": 10, 
    "phase_size": 4
}
critic_hyperparameters = {
    "h_size": [256, 64, 16],
    "learning_rate": 0.01
}
actor_hyperparameters = {
    "reader_h_size": 16,
    "h_size": [256, 64, 16],
    "action_size": 1,
    "learning_rate": 0.01
}

num_epsiodes = 600

Grid9 = TrafficEnv('./networks/data/Grid9.sumocfg', parameters["log_dir"], gui=False)
Agent = MA_MDDPG(Grid9, parameters, encoder_hyperparameters, critic_hyperparameters, actor_hyperparameters)
for i in range(num_epsiodes):
    Agent.step()
    print("episode:%d  total_reward:%.2f" % (i, Agent.total_reward_per_epsiode[-1]))
    if i % 200 == 0 or 590 - i <= 0:
        Agent.save_model()
Agent.env.output_data()
