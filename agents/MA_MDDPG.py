import torch
import torch.nn.functional as functional
from torch import optim
from utilities.ReplayBuffer import Replay_Buffer
from MemoryBasedDDPG import MemoryDDPG

class MA_MDDPG():
    def __init__(self, env):
        self.env = env
        