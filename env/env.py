import os
import logging
import numpy as np
from sumolib import checkBinary
import pandas as pd
import traci
import traci.constants as tc
import xml.etree.cElementTree as ET

ILD_LENGTH = 100
VER_LENGTH = 5
# N_LANES = 3
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
WIND_MAP = {'I0':{'P0':0, 'I3':0, 'P6':3, 'I1':3},
            'I1':{'P1':0, 'I4':0, 'I0':3, 'I2':3},
            'I2':{'P2':0, 'I5':0, 'I1':3, 'P9':3},
            'I3':{'I0':0, 'I6':0, 'P7':3, 'I4':3},
            'I4':{'I1':0, 'I7':0, 'I3':3, 'I5':3},
            'I5':{'I2':0, 'I8':0, 'I4':3, 'P10':3},
            'I6':{'I3':0, 'P3':0, 'P8':3, 'I7':3},
            'I7':{'I4':0, 'P4':0, 'I6':3, 'I8':3},
            'I8':{'I5':0, 'P5':0, 'I7':3, 'P11':3}}

class TrafficNode:
    def __init__(self, name, neighbor=[]):
        self.name = name
        self.neighbor = neighbor
        self.lanes_in = []
        self.ilds_in = []
        # self.phase_id = -1 
    



class TrafficEnv:
    def __init__(self, cfg_sumo, output_path='./logs/', port=4343, gui=False):
        self.cfg_sumo = cfg_sumo
        self.port = port
        self.cur_episode = 0
        self.margin = 13.6
        self.neighbor_map = NEIGHBOR_MAP
        self.phase_map = PHASE_MAP
        self.ild_length = ILD_LENGTH
        self.ver_length = VER_LENGTH
        self.wind_map = WIND_MAP

        self.sim_seed = 42
        self.name = 'Grid9'
        self.agent = 'ma2c'
        self.output_path = output_path
        self.control_interval_sec = 5
        self.yellow_interval_sec = 2
        self.episode_length_sec = 3600
        self.coef_reward = 0.1

        # self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        
        # params need reset
        self.cur_step = 0

        # if not os.path.exists(self.output_path+'/logs'):
        #     os.makedirs(self.output_path+'./logs')
        self.metric_data = []
        self.step_data = []
        # self.metrics_file = self.output_path + 'metrics.csv'
        # with open(self.metrics_file, 'w') as f:
        #     f.write('episode,time,step,number_total_car,number_departed_car,number_arrived_car,avg_wait_sec,avg_speed_mps,avg_queue\n')
        # self.step_file = self.output_path + 'step.csv'
        # with open(self.step_file, 'w') as f:
        #     f.write('episode,time,step,action,reward_jam,reward_waiting,reward,total_reward\n')

        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), "--start", '-c', self.cfg_sumo]
        command += ['--seed', str(self.sim_seed)]
        command += ['--no-step-log', 'True']
        command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # command += ['--tripinfo-output',
        #             self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        traci.start(command, port=self.port)
        self.nodes = self._init_node()
        self.nodes_name = sorted(list(self.nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.nodes_name)
        for node_name in self.nodes_name:
            s += node_name + ':\n'
            s += '\tneigbor: %s\n' % str(self.nodes[node_name].neighbor) 
        logging.info(s)
        for node_name in self.nodes_name:
            traci.junction.subscribeContext(node_name, tc.CMD_GET_VEHICLE_VARIABLE, self.ild_length,
                                            [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION,
                                            tc.VAR_SPEED, tc.VAR_WAITING_TIME])
        

    def _init_node(self):
        nodes = {}
        for node_name in traci.trafficlight.getIDList():
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found' % node_name)
                neighbor = []
            nodes[node_name] = TrafficNode(node_name, neighbor)
            nodes[node_name].lanes_in = traci.trafficlight.getControlledLanes(node_name)
            nodes[node_name].ilds_in = nodes[node_name].lanes_in
        return nodes

    def _get_obs(self, cx_res):
        height = int(self.ild_length/self.ver_length)
        position, phase = {}, {}      
        for node_name in self.nodes_name:
            width = int(len(self.nodes[node_name].lanes_in)/2) - 2
            # print(self.nodes[node_name].lanes_in)
            position[node_name] = np.zeros(shape=(height, width))
            phase[node_name] = np.zeros(shape=(int(len(self.phase_map)/2)))
            current_phase = int(traci.trafficlight.getPhase(node_name)/2)
            phase[node_name][current_phase] = 1
        if not cx_res:
            return [position, phase]
        for node_name, res in cx_res.items():
            if not res:
                continue
            for _, mes in res.items():
                f_node, t_node, lane = mes[tc.VAR_LANE_ID].split('_')
                if t_node == node_name:
                    wind = self._get_position_windex(f_node, t_node, lane)
                    if f_node[0] == 'I':
                        hind = int((500 - 2 * self.margin - mes[tc.VAR_LANEPOSITION]) / self.ver_length)
                    elif f_node[0] == 'P':
                        hind = int((200 - self.margin - mes[tc.VAR_LANEPOSITION]) / self.ver_length)
                    if hind < 0 or hind >= height:
                        logging.info(str(res))
                        raise ValueError(str(hind)+'  h_ind is wrong')
                    position[node_name][hind, wind] += 1
            if np.amax(position[node_name]) > 2:
                raise ValueError('max value of position need <= 2')
        return [position, phase]
    
    def _get_position_windex(self, from_node, to_node, n_lane):
        return int(n_lane) + self.wind_map[to_node].get(from_node)
    
    def _get_reward(self, cx_res, action):
        reward = {}
        reward_jam = {}
        reward_waiting = {}
        for node_name in self.nodes_name:
            if not cx_res:
                reward[node_name], reward_jam[node_name], reward_waiting[node_name] = 0, 0, 0
                continue
            res = cx_res.get(node_name)
            if res is None:
                reward[node_name], reward_jam[node_name], reward_waiting[node_name] = 0, 0, 0
                continue
            jam_length, waitingtime = 0, 0
            for ild in self.nodes[node_name].ilds_in:
                jam_length += traci.lanearea.getJamLengthVehicle(ild)
            for _, mes in res.items():
                _, t_node, _ = mes[tc.VAR_LANE_ID].split('_')
                if t_node == node_name:
                    waitingtime += mes[tc.VAR_WAITING_TIME]
            reward_jam[node_name] = -jam_length
            reward_waiting[node_name] = -waitingtime
            reward[node_name] = -jam_length - self.coef_reward * waitingtime
        return reward, reward_jam, reward_waiting
    
    def _measure_step(self):
        cars = traci.vehicle.getIDList()
        num_tot_car = len(cars)
        num_in_car = traci.simulation.getDepartedNumber()
        num_out_car = traci.simulation.getArrivedNumber()
        if num_tot_car > 0:
            avg_waiting_time = np.mean([traci.vehicle.getWaitingTime(car) for car in cars])
            avg_speed = np.mean([traci.vehicle.getSpeed(car) for car in cars])
        else:
            avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []
        for node_name in self.nodes_name:
            for ild in self.nodes[node_name].ilds_in:
                queues.append(traci.lane.getLastStepHaltingNumber(ild))
        avg_queue = np.mean(np.array(queues))
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_step,
                       'step': self.cur_step / self.control_interval_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'avg_queue': avg_queue}
        self.metric_data.append(cur_traffic)
    
    def _simulate(self, num_steps):
        for _ in range(num_steps):
            traci.simulationStep()
            self.cur_step += 1
        
    
    def step(self, action):
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
        self._measure_step()
        cx_res = {node_name: traci.junction.getContextSubscriptionResults(node_name) \
                  for node_name in self.nodes_name}
        obs = self._get_obs(cx_res)
        reward, reward_jam, reward_waiting = self._get_reward(cx_res, action)
        done = True if self.cur_step >= self.episode_length_sec else False
        info = {'episode': self.cur_episode, 
                'time': self.cur_step,
                'step': self.cur_step / self.control_interval_sec,
                'action': [action[node_name] for node_name in self.nodes_name],
                'reward_jam':[reward_jam[node_name] for node_name in self.nodes_name],
                'reward_waiting':[reward_waiting[node_name] for node_name in self.nodes_name],
                'reward': [reward[node_name] for node_name in self.nodes_name],
                'total_reward': np.sum([reward[node_name] for node_name in self.nodes_name])}
        self.step_data.append(info)
        return obs, reward, done, info
    
    def reset(self, gui=False):
        # return obs
        # for node_name in self.nodes_name:
        #     self.nodes[node_name].reset()
        self.cur_episode += 1
        self.cur_step = 0
        # self.close()
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = ['--start','-c', self.cfg_sumo]
        # command += ['--seed', str(self.sim_seed)]
        # command += ['--no-step-log', 'True']
        # command += ['--time-to-teleport', '300']
        # command += ['--no-warnings', 'True']
        # command += ['--duration-log.disable', 'True']
        # command += ['--tripinfo-output',
        #             self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        traci.load(command)
        s = 'Env: init %d node information:\n' % len(self.nodes_name)
        for node_name in self.nodes_name:
            s += node_name + ':\n'
            s += '\tneigbor: %s\n' % str(self.nodes[node_name].neighbor) 
        logging.info(s)
        for node_name in self.nodes_name:
            traci.junction.subscribeContext(node_name, tc.CMD_GET_VEHICLE_VARIABLE, self.ild_length,
                                            [tc.VAR_LANE_ID, tc.VAR_LANEPOSITION,
                                            tc.VAR_SPEED, tc.VAR_WAITING_TIME])
        cx_res = {node_name: traci.junction.getContextSubscriptionResults(node_name) \
                  for node_name in self.nodes_name}
        return self._get_obs(cx_res)
    
    def close(self):
        traci.close()
    
    def seed(self, seed=None):
        if seed:
            np.random.seed(seed)

    def output_data(self):
        step_data = pd.DataFrame(self.step_data)
        step_data.to_csv(self.output_path  +  ('%s_%s_step_%d.csv' % (self.name, self.agent, self.cur_episode)))
        metric_data = pd.DataFrame(self.metric_data)
        metric_data.to_csv(self.output_path  + ('%s_%s_metric_%d.csv' % (self.name, self.agent, self.cur_episode)))

if __name__ =='__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.WARNING)
    Grid9 = TrafficEnv('../networks/data/Grid9.sumocfg', gui=True)
    action = {}
    import random
    while True:
        for node_name in Grid9.nodes_name:
            action[node_name] = random.randint(0,1)
        _, _, done, _ = Grid9.step(action)
        if done:
            break

    
    # logging.info("step 0")
    # Grid9.step(action)
    # logging.info("step 1")
    # Grid9.step(action)
    # logging.info("step 2")
    # Grid9.step(action)
    # logging.info("step 3")
    # Grid9.step(action)
    # logging.info("step 4")
    # Grid9.step(action)
    # logging.info("reset")
    # Grid9.reset(gui=True)
    # logging.info("step 0")
    # Grid9.step(action)
    # logging.info("step 1")
    # Grid9.step(action)
    # logging.info("step 2")
    # Grid9.step(action)
    # logging.info("step 3")
    # Grid9.step(action)
    # logging.info("step 4")
    # Grid9.step(action)
    Grid9.output_data()
    Grid9.close()
    
