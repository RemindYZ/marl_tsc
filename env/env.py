import logging
import numpy as np
from sumolib import checkBinary
# import subprocess
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET

LENGTH = 100
NEIGHBOR_MAP = {'I0':['I1', 'I3'],
                'I1':['I0', 'I2', 'I4'],
                'I2':['I1', 'I5'],
                'I3':['I0', 'I4', 'I6'],
                'I4':['I1', 'I3', 'I5', 'I7'],
                'I5':['I2', 'I4', 'I8'],
                'I6':['I3', 'I7'],
                'I7':['I4', 'I6', 'I8'],
                'I8':['I5', 'I7']}
PHASE_MAP = {0:'GGGrrrrrGGGrrrrr', 1:'yyyrrrrryyyrrrrr',
             2:'rrrGrrrrrrrGrrrr', 3:'rrryrrrrrrryrrrr',
             4:'rrrrGGGrrrrrGGGr', 5:'rrrryyyrrrrryyyr',
             6:'rrrrrrrGrrrrrrrG', 7:'rrrrrrryrrrrrrry'}

class TrafficNode:
    def __init__(self, name, dim_memory, neighbor=[]):
        self.name = name
        self.dim_memory = dim_memory
        self.neighbor = neighbor
        self.lanes_in = []
        self.ilds_in = []
        self.phase_id = -1 
        self.memory = np.random.randn(self.dim_memory)
    
    def write(self, mes):
        self.memory = mes
    
    def read(self):
        return self.memory
    
    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)
        self.memory = np.random.randn(self.dim_memory)


class TrafficEnv:
    def __init__(self, cfg_sumo, cfg_param, port=None, gui=False):
        self.cfg_sumo = cfg_sumo
        self.port = port
        self.cur_episode = 0
        self.neighbor_map = NEIGHBOR_MAP
        self.phase_map = PHASE_MAP
        self.length = LENGTH

        # params from config
        self.sim_seed = cfg_param.getint('sim_seed')
        self.name = cfg_param.get('name')
        self.agent = cfg_param.get('agent')
        self.n_junction = cfg_param.get('n_junction') # 9
        self.dim_memory = cfg_param.get('dim_memory')
        self.output_path = cfg_param.get('output_path')
        self.control_interval_sec = cfg_param.getint('control_interval_sec') # 5s
        self.yellow_interval_sec = cfg_param.getint('yellow_interval_sec') # 2s
        self.episode_length_sec = cfg_param.getint('episode_length_sec') # 3600s

        # self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        
        # params need reset
        self.cur_step = 0
        
        
        self.nodes = self._init_node()
        self.nodes_name = sorted(list(self.nodes.keys()))

        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', self.cfg_sumo]
        command += ['--seed', str(self.sim_seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--tripinfo-output',
                    self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        traci.start(command, port=self.port)
        for i in range(self.n_junction):
            traci.junction.subscribeContext('I'+str(i), tc.CMD_GET_VEHICLE_VARIABLE, self.length,
                                            [tc.VAR_LANE_ID, tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION,
                                            tc.VAR_SPEED, tc.VAR_WAITING_TIME])

    def _init_node(self):
        nodes = {}
        for node_name in traci.trafficlight.getIDList():
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found' % node_name)
                neighbor = []
            nodes[node_name] = TrafficNode(node_name, self.dim_memory, neighbor)
            nodes[node_name].lanes_in = traci.trafficlight.getControlledLanes(node_name)
            nodes[node_name].ilds_in = nodes[node_name].lanes_in
        return nodes

    def _get_obs(self, cx_res):
        obs = None
        return obs
    
    def _get_reward(self, cx_res, action):
        reward = None
        return reward
    
    def _measure_step(self):
        pass
    
    def _simulate(self, num_steps):
        for _ in range(num_steps):
            traci.simulationStep()
            self.cur_step += 1
            self._measure_step()
        
    
    def step(self, action):
        # return obs, reward, done, info
        for node_name in self.nodes_name:
            a = action[node_name]
            current_phase = traci.trafficlight.getPhase(node_name)
            next_phase = (current_phase + a) % len(self.phase_map)
            traci.trafficlight.setPhase(node_name, next_phase)
        self._simulate(self.yellow_interval_sec)
        for node_name in self.nodes_name:
            a = action[node_name]
            current_phase = traci.trafficlight.getPhase(node_name)
            next_phase = (current_phase + a) % len(self.phase_map)
            traci.trafficlight.setPhase(node_name, next_phase)
        self._simulate(self.control_interval_sec-self.yellow_interval_sec)

        cx_res = {node_name: traci.junction.getContextSubscriptionResults(node_name) \
                  for node_name in self.nodes_name}
        obs = self._get_obs(cx_res)
        reward = self._get_reward(cx_res, action)
        done = True if self.cur_step >= self.episode_length_sec else False
        info = {'episode': self.cur_episode, 
                'time': self.cur_step,
                'action': action,
                'reward': reward}
        return obs, reward, done, info
    
    def reset(self):
        # return obs
        self.cur_episode += 1
        pass
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)
    
