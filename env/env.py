import logging
import numpy as np
from sumolib import checkBinary
import subprocess
import random
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET



class TrafficNode:
    def __init__(self, name, neighbor=[]):
        pass

class TrafficEnv:
    def __init__(self, cfg_sumo, cfg_param, port=None, gui=False):
        self.cfg_sumo = cfg_sumo
        self.seed = cfg_param.getint('seed')
        self.port = port
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', self.cfg_sumo]
        command += ['--seed', str(self.seed)]
        command += ['--remote-port', str(self.port)]

        
    
    def step(self, action):
        # return obs, reward, done, info
        pass
    
    def reset(self):
        # return obs
        pass
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        if seed:
            random.seed(seed)
