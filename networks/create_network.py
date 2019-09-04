import numpy as np
import os
import random

PATH = './data/'
INTERSECTION_DISTANCE = 500
END_DISTANCE = 200
N_INTERSECTION = 3
SPEED_LIMIT = 16.67

class NetworkGenerator():
    def __init__(self, name_network):
        self.name_network = name_network
        self.path = PATH
        self.i_distance = INTERSECTION_DISTANCE
        self.e_distance = END_DISTANCE
        self.n_intersection = N_INTERSECTION
    
    def create_network(self, init_density, seed=None, thread=None):
        self.gen_nod_file()
        self.gen_typ_file()
        self.gen_edg_file()
        self.gen_con_file()
        self.gen_tll_file()
        self.gen_net_file()
        self.gen_rou_file(init_density, seed)
        self.gen_add_file()
        self.gen_sumocfg(thread)

    def _write_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

    def gen_nod_file(self):
        path = self.path + self.name_network +'.nod.xml'
        node_context = '<nodes>\n'
        node_str = '  <node id="%s" x="%.2f" y="%.2f" type="%s"/>\n'
        length = self.i_distance*self.n_intersection
        index = 0
        for y in range(0, length, self.i_distance):
            for x in range(0, length, self.i_distance):
                node_context += node_str % ('I' + str(index), x, y, 'traffic_light')
                index += 1
        index = 0
        for x in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), x, -self.e_distance, 'priority')
            index += 1
        for x in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), x, length-self.i_distance+self.e_distance, 'priority')
            index += 1
        for y in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), -self.e_distance, y, 'priority')
            index += 1
        for y in range(0, length, self.i_distance):
            node_context += node_str % ('P' + str(index), length-self.i_distance+self.e_distance, y, 'priority')
            index += 1
        node_context += '</nodes>\n'
        self._write_file(path, node_context)

    def gen_typ_file(self):
        path = self.path + self.name_network +'.typ.xml'
        type_context = '<types>\n'
        type_context += '  <type id="a" numLanes="3" speed="%.2f"/>\n' % SPEED_LIMIT
        type_context += '</types>\n'
        self._write_file(path, type_context)
    
    def _gen_edg_str(self, edge_str, from_node, to_node, edge_type):
        edge_id = 'e:%s_%s' %(from_node, to_node)
        return edge_str %(edge_id, from_node, to_node, edge_type)
    
    def gen_edg_file(self):
        path = self.path + self.name_network +'.edg.xml'
        edges_context = '<nodes>\n'
        edges_str = '  <edge id="%s" from="%s" to="%s" type="%s"/>\n'
        node_pair = [('P0','I0'),('P1','I1'),('P2','I2'),
                    ('P3','I6'),('P4','I7'),('P5','I8'),
                    ('P6','I0'),('P7','I3'),('P8','I6'),
                    ('P9','I2'),('P10','I5'),('P11','I8'),
                    ('I0','I1'),('I1','I2'),('I3','I4'),('I4','I5'),('I6','I7'),('I7','I8'),
                    ('I0','I3'),('I1','I4'),('I2','I5'),
                    ('I3','I6'),('I4','I7'),('I5','I8')]
        for (i1,i2) in node_pair:
            edges_context += self._gen_edg_str(edges_str, i1, i2, 'a')
            edges_context += self._gen_edg_str(edges_str, i2, i1, 'a')
        edges_context += '</edges>\n'
        self._write_file(path, edges_context)
    
    def _gen_con_str(self, con_str, from_node, cur_node, to_node, from_lane, to_lane):
        from_edge = 'e:%s_%s' % (from_node, cur_node)
        to_edge = 'e:%s_%s' % (cur_node, to_node)
        return con_str % (from_edge, to_edge, from_lane, to_lane)
    
    def _gen_con_node(self, con_str, cur_node, n_node, s_node, w_node, e_node):
        str_cons = ''
        # go-through
        str_cons += self._gen_con_str(con_str, s_node, cur_node, n_node, 0, 0)
        str_cons += self._gen_con_str(con_str, n_node, cur_node, s_node, 0, 0)
        str_cons += self._gen_con_str(con_str, s_node, cur_node, n_node, 1, 1)
        str_cons += self._gen_con_str(con_str, n_node, cur_node, s_node, 1, 1)
        str_cons += self._gen_con_str(con_str, w_node, cur_node, e_node, 0, 0)
        str_cons += self._gen_con_str(con_str, e_node, cur_node, w_node, 0, 0)
        str_cons += self._gen_con_str(con_str, w_node, cur_node, e_node, 1, 1)
        str_cons += self._gen_con_str(con_str, e_node, cur_node, w_node, 1, 1)
        # left-turn
        str_cons += self._gen_con_str(con_str, s_node, cur_node, w_node, 2, 1)
        str_cons += self._gen_con_str(con_str, n_node, cur_node, e_node, 2, 1)
        str_cons += self._gen_con_str(con_str, w_node, cur_node, n_node, 2, 1)
        str_cons += self._gen_con_str(con_str, e_node, cur_node, s_node, 2, 1)
        # right-turn
        str_cons += self._gen_con_str(con_str, s_node, cur_node, e_node, 0, 0)
        str_cons += self._gen_con_str(con_str, n_node, cur_node, w_node, 0, 0)
        str_cons += self._gen_con_str(con_str, w_node, cur_node, s_node, 0, 0)
        str_cons += self._gen_con_str(con_str, e_node, cur_node, n_node, 0, 0)
        return str_cons
    
    def gen_con_file(self):
        path = self.path + self.name_network +'.con.xml'
        connections_context = '<connections>\n'
        connections_str = '  <connection from="%s" to="%s" fromLane="%d" toLane="%d"/>\n'
        node_pair = [('I0','I3','P0','P6','I0'),('I1','I4','P1','I0','I2'),('I2','I5','P2','I1','P9'),
                    ('I3','I6','I0','P7','I4'),('I4','I7','I1','I3','I5'),('I5','I8','I2','I4','P10'),
                    ('I6','P3','I3','P8','I7'),('I7','P4','I4','I6','I8'),('I8','P5','I5','I7','P11')]
        for (cur,n,s,w,e) in node_pair:
            connections_context += self._gen_con_node(connections_str, cur, n, s, w, e)
        connections_context += '</connections>\n'
        self._write_file(path, connections_context)

    def gen_tll_file(self):
        random.seed()
        path = self.path + self.name_network +'.tll.xml'
        tls_str = '  <tlLogic id="%s" programID="0" offset="%d" type="static">\n'
        phase_str = '    <phase duration="%d" state="%s"/>\n'
        tls_context = '<additional>\n'
        phases = [('GGrrrrGGrrrr',25), ('yyrrrryyrrrr',3),
                 ('rrGrrrrrGrrr',15), ('rryrrrrryrrr',3),
                 ('rrrGGrrrrGGr',25), ('rrryyrrrryyr',3),
                 ('rrrrrGrrrrrG',15), ('rrrrryrrrrry',3)]
        for ind in range(self.n_intersection*self.n_intersection):
            offset = random.randint(0,91)
            node_id = 'I' + str(ind)
            tls_context += tls_str % (node_id, offset)
            for (state, duration) in phases:
                tls_context += phase_str % (duration, state)
            tls_context += '  </tlLogic>\n'
        tls_context += '</additional>\n'
        self._write_file(path, tls_context)
    
    def gen_net_file(self):
        config_context = '<configuration>\n  <input>\n'
        config_context += '    <edge-files value="exp.edg.xml"/>\n'
        config_context += '    <node-files value="exp.nod.xml"/>\n'
        config_context += '    <type-files value="exp.typ.xml"/>\n'
        config_context += '    <tllogic-files value="exp.tll.xml"/>\n'
        config_context += '    <connection-files value="exp.con.xml"/>\n'
        config_context += '  </input>\n  <output>\n'
        config_context += '    <output-file value="exp.net.xml"/>\n'
        config_context += '  </output>\n</configuration>\n'
        path = self.path + self.name_network +'.netccfg'
        self._write_file(path, config_context)
        os.system('netconvert -c '+ self.name_network + '.netccfg')
    
    def gen_rou_file(self, init_density, seed=None):
        # if seed is not None:
        #     random.seed(seed)
        # ext_flow = '  <flow id="f:%s" departPos="random_free" from="%s" to="%s" begin="%d" end="%d" vehsPerHour="%d" type="type1"/>\n'
        # str_flows = '<routes>\n'
        # str_flows += '  <vType id="type1" length="5" accel="2.6" decel="4.5"/>\n'
        pass
    
    def _gen_add_str(self, ild_str, from_node, to_node, n_lane):
        edge_id = '%s_%s' % (from_node, to_node)
        edge_add_str = ''
        for i in range(n_lane):
            edge_add_str += ild_str % (edge_id, i, edge_id, i)
        return edge_add_str
    
    def _gen_add_node(self, ild_str, cur_node, n_node, s_node, w_node, e_node):
        node_add_str = ''
        node_add_str += self._gen_add_str(ild_str, n_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, s_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, w_node, cur_node, n_lane=3)
        node_add_str += self._gen_add_str(ild_str, e_node, cur_node, n_lane=3)
        return node_add_str

    def gen_add_file(self):
        path = self.path + self.name_network +'.add.xml'
        ild_context = '<additional>\n'
        ild_str = '  <laneAreaDetector file="ild.out" freq="1" id="ild:%s_%d" lane="e:%s_%d"\
             pos="-100" endPos="-1"/>\n'
        node_pair = [('I0','I3','P0','P6','I0'),('I1','I4','P1','I0','I2'),('I2','I5','P2','I1','P9'),
                    ('I3','I6','I0','P7','I4'),('I4','I7','I1','I3','I5'),('I5','I8','I2','I4','P10'),
                    ('I6','P3','I3','P8','I7'),('I7','P4','I4','I6','I8'),('I8','P5','I5','I7','P11')]
        for (cur,n,s,w,e) in node_pair:
            ild_context += self._gen_con_node(ild_str, cur, n, s, w, e)
        ild_context += '</additional>\n'
        self._write_file(path, ild_context)
    
    def gen_sumocfg(self, thread=None):
        path = self.path + self.name_network +'.sumocfg'
        if thread is None:
            out_file = 'exp.rou.xml'
        else:
            out_file = 'exp_%d.rou.xml' % int(thread)
        config_context = '<configuration>\n  <input>\n'
        config_context += '    <net-file value="exp.net.xml"/>\n'
        config_context += '    <route-files value="%s"/>\n' % out_file
        config_context += '    <additional-files value="exp.add.xml"/>\n'
        config_context += '  </input>\n  <time>\n'
        config_context += '    <begin value="0"/>\n    <end value="3600"/>\n'
        config_context += '  </time>\n</configuration>\n'
        self._write_file(path, config_context)

if __name__=='__main__':
    ng = NetworkGenerator('Grid9')
    ng.create_network()