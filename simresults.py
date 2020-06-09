import os
from collections import namedtuple
import numpy as np
import scipy.interpolate

class SimResults(object):
    def __init__(self, sim_folder):
        if sim_folder == '':
            sim_folder=os.getcwd()
        self.sim_folder=sim_folder
        file_list = os.listdir(sim_folder)
        if 'spcforc' in file_list:
            self.spcforc = readspcforc(sim_folder)
        if 'nodout' in file_list:
            self.nodout = readnodout(sim_folder)
            self.time=list(self.nodout.values())[0].time
        if 'bndout' in file_list:
            self.bndout = readbndout(sim_folder)
        if 'glstat' in file_list:
            self.glstat = readglstat(sim_folder)
        if 'rwforc' in file_list:
            self.rwforc = readrwforc(sim_folder)
        if 'rbdout' in file_list:
            self.rbout = readrbdout(sim_folder)
        if 'nodfor' in file_list:
            self.nodfor = readnodfor(sim_folder)

        self.dX_ref = None

    def PK1(self, source='bndout'):
        if source == 'bndout':
            time=list(self.bndout.values())[0].time
            traction = traction_bndout(self.bndout, node_order_cube(self.nodout))

        elif source == 'rcforc':
            time = list(self.rcforc.values())[0].time
            traction = traction_rcforc(self.rcforc, node_order_cube(self.nodout))

        PK1 = traction / self.A0()
        PK1_interp = scipy.interpolate.interp1d(time, PK1, axis=0)
        return PK1_interp(self.time)

    def F(self):
        F_interp, _ = def_gradient(self.nodout)
        return F_interp(self.time)

    def inf_strain(self):
        F_temp=self.F()
        return 0.5*(np.transpose(F_temp, (0, 2, 1))+F_temp)-np.identity(3)

    def A0(self):
        F_interp, dX = def_gradient(self.nodout, dX_ref=self.dX_ref)
        A0 = area(F_interp(0.0), dX)
        return A0
    # os.chdir(workingFolder)

def readrwforc(sim_folder):
    '''Reads timestamps and resultant forces on rwforc file
    returns dict with each wall in tuple [t, R_xyz)]
    NOT TESTED'''
    rw_tup = namedtuple('rwforc', ['time', 'R'])
    with open(r'{}\rwforc'.format(sim_folder)) as rwforc:
        lines = rwforc.readlines()
    data_start = [index for index, line in enumerate(lines) if 'time       wall#' in line][0] + 1
    formated_data = np.array([[float(line.split()[0]), int(line.split()[1]), float(line.split()[2]),
                              float(line.split()[3]), float(line.split()[4]),
                              float(line.split()[5].replace('\n', ''))
                              ] for line in lines[data_start:]])
    wall_dict = {}
    for wall_id in set(formated_data[:, 1]):
        wall_index = np.where(formated_data[:, 1] == wall_id)
        wall_dict[int(wall_id)] = rw_tup(formated_data[wall_index, 0][0], np.array([formated_data[wall_index, 3][0],
                                      formated_data[wall_index, 4][0], formated_data[wall_index, 5][0]]))
    return wall_dict

def readrcforc(sim_folder):
    '''Reads timestamps and resultant forces on rcforc file
    returns dict of reaction forces with tuple [t, R_xyz]'''
    rc_tup = namedtuple('rcforc', ['time', 'R'])
    with open(r'{}\rcforc'.format(sim_folder)) as rcforc:
        lines = rcforc.readlines()

    force_trans_id=[int(line.split()[0]) for line in lines if 'Force Transducer' in line]
    resultant_dict=[]
    for ftid in force_trans_id:
        time = []
        RX = []
        RY = []
        RZ = []
        for line in lines:
            temp=line.split()
            if len(temp) > 2:
                if temp[0]=='slave' and int(temp[1]) == ftid:
                    time.append(float(temp[3]))
                    RX.append(float(temp[5]))
                    RY.append(float(temp[7]))
                    RZ.append(float(temp[9]))

        resultant_dict[ftid] =rc_tup(np.array(time), np.array([RX, RY, RZ]))
    return resultant_dict

def readrbdout(sim_folder):
    '''Reads timestamps and resultant forces on rwforc file
    returns [t, x,y,z]'''
    rwTup = namedtuple('rbdoutTup', ['time', 'R'])
    with open(r'{}\rbdout'.format(sim_folder)) as rwforc:
        lines = rwforc.readlines()
    dataStart = [index for index, line in enumerate(lines) if 'time       wall#' in line][0] + 1
    formatedData = np.array([[float(line.split()[0]), int(line.split()[1]), float(line.split()[2]),
                              float(line.split()[3]), float(line.split()[4]),
                              float(line.split()[5].replace('\n', ''))
                              ] for line in lines[dataStart:]])
    wallDict = {}
    for wallID in set(formatedData[:, 1]):
        wallIndex = np.where(formatedData[:, 1] == wallID)
        wallDict[int(wallID)] = rwTup(formatedData[wallIndex, 0], formatedData[wallIndex, 3:6])
    return wallDict

def readspcforc(sim_folder):
    '''Reads timestamps and resultant forces on spc file
    returns [t],[x,y,z]'''
    #Filter by heading? Not tried
    spcforc_tup = namedtuple('spcTup', ['time', 'R'])
    with open(r'{}\spcforc'.format(sim_folder)) as spcforc:
        lines = spcforc.readlines()
    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if 'output at time' in line])
    if len(output_time_data) == 0:
        return []
    timestamps = output_time_data[:, 1]
    output_time_ind = output_time_data[:, 0]

    force_resultant_ind = output_time_ind + output_time_ind[1] - output_time_ind[0] - 1

    force_resultant = np.array([[float(item) for item in lines[int(ind)].split()[-3:]] for ind in force_resultant_ind])

    return spcforc_tup(timestamps, force_resultant[:, 0:3])

def readbndout(sim_folder):
    '''Reads timestamps and resultant forces on bnd file
    returns [t],F_xyz'''
    bnd_tup = namedtuple('bndTup', ['time', 'F'])
    with open(r'{}\bndout'.format(sim_folder)) as bndout:
        lines = bndout.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if r' t= ' in line])
    if len(output_time_data) == 0:
        return []
    timestamps = output_time_data[:, 1]
    output_time_ind = output_time_data[:, 0]
    numnodes = int((output_time_ind[1] - output_time_ind[0]) - 6)
    header = 4
    node_values = [[] for _ in range(numnodes)]
    for ind, time in zip(output_time_ind, timestamps):
        for i in range(numnodes):
            line = lines[int(ind) + header + i]
            nid = int(line.split()[1])
            values = [nid] + [float(force) for force in line.split()[3:8:2]]
            node_values[i].append(values)
    bndout_dict = {}
    for j in range(numnodes):
        values = np.array(node_values[j])
        bndout_dict[int(values[0, 0])] = bnd_tup(timestamps, values[:, 1:4])
    return bndout_dict#, node_order

def readglstat(sim_folder):
    '''Reads timestamps and resultant forces on bnd file
    returns [t],[x,y,z]'''
    glstat_tup = namedtuple('glstatTup', ['time', 'time_step', 'kinetic_energy', 'internal_energy',
                                         'external_work', 'total_energy', 'energy_ratio', 'mass_increase_percent'])
    with open(r'{}\glstat'.format(sim_folder)) as glstat:
        lines = glstat.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if 'time......' in line])
    if len(output_time_data) == 0:
        return []
    timestamps = output_time_data[:, 1]
    output_time_ind = output_time_data[:, 0]
    value_ind = output_time_ind
    values = []
    value_keys = {0: 1, 1: 2, 2: 3, 3: 7, 4: 11, 5: 13, 6: 25}
    for i in range(7):
        tempInd = value_ind + value_keys[i]
        try:
            values.append([float(lines[int(ind)].split()[-1]) for ind in tempInd])
        except:
            values.append([None])
    return glstat_tup(timestamps, *values)

def readnodout(sim_folder):
    '''Reads timestamps and resultant forces on nodoutfile
    returns [t],[x,y,z]'''
    nod_tup = namedtuple('nodTup', ['time', 'D', 'V', 'A'])
    with open(r'{}\nodout'.format(sim_folder)) as nodout:
        lines = nodout.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-2])] for i, line in enumerate(lines) if 'at time' in line][::2])
    if len(output_time_data) == 0:
        return []
    timestamps = output_time_data[:, 1]
    output_time_ind = output_time_data[:, 0]
    numnodes = int((output_time_ind[1] - output_time_ind[0]) / 2 - 6)
    header = 3
    node_values = [[] for _ in range(numnodes)]
    for ind, time in zip(output_time_ind, timestamps):
        for i in range(numnodes):
            line = lines[int(ind) + header + i]
            nid = int(line[:10])
            line = line[10:].replace('\n', '')
            values = [nid] + list(map(float, map(''.join, zip(*[iter(line)] * 12))))
            node_values[i].append(values)
    nodout_dict = {}
    for i in range(numnodes):
        values = np.array(node_values[i])
        nodout_dict[int(values[0, 0])] = nod_tup(timestamps, values[:, 1:4], values[:, 4:7], values[:, 10:13])
    return nodout_dict

def readnodfor(sim_folder):
    '''Reads timestamps and resultant forces on nodfor file
    returns list of [t],[x,y,z]'''
    nodfor_tup = namedtuple('nodfor_tup', ['time', 'R'])
    with open(r'{}\nodfor'.format(sim_folder)) as nodfor:
        lines = nodfor.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if
         'n o d a l   f o r c e   g r o u p    o u t p u t  t' in line][:])
    if len(output_time_data) == 0:
        return []
    timestamps = output_time_data[:, 1]
    output_time_ind = list(map(int, output_time_data[:, 0]))
    numnodes = output_time_ind[1] - output_time_ind[0]
    force_values = {}
    ##
    #Map set_id to group_id?
    group_to_setid={}
    for line in lines[:output_time_ind[0]]:
        if 'Group from set' in line:
            line = line.split()
            group_to_setid[int(line[6])] = int(line[0])

    for ind, time in zip(output_time_ind, timestamps):
        for line in lines[ind:(ind + numnodes)]:
            if 'nodal group output number' in line:
                group_id = int(line.split()[-1])
            if 'xtotal' in line:
                line = line.replace('\n', '')
                line = line.split()
                if ind==output_time_ind[0]:
                    force_values[group_id] = []
                force_values[group_id].append([float(value) for value in line[1:-1:2]])
    force_dict = {}
    for node_group in force_values.items():
        force_dict[group_to_setid[node_group[0]]] = nodfor_tup(timestamps, np.array(node_group[1]))
    return force_dict

def node_order_cube(nodout_dict):
    '''Z-axis point up, second node along x-axis'''
    corner_nodes = list(range(8))
    loc_bools = [[True, True, True],
                         [False, True, True],
                         [False, False, True],
                         [True, False, True],
                         [True, True, False],
                         [False, True, False],
                         [False, False, False],
                         [True, False, False]]
    for node in nodout_dict.items():
        for ind, loc_bool in enumerate(loc_bools):
            if all((node[1].A[0] == [0, 0, 0]) == loc_bool):
                corner_nodes[ind] = node[0]
    return corner_nodes

def def_gradient(nodout_dict, dX_ref=None):
    node_order=node_order_cube(nodout_dict)
    dX = np.array([nodout_dict[node].A[0] for node in node_order])
    if dX_ref != None:
        dX.astype(bool).astype(int)
        dX=dX*dX_ref
    # deformed vectors
    dx = np.array([nodout_dict[node].A for node in node_order]).swapaxes(0, 1)
    time = nodout_dict[node_order[0]].time
    #dx_interp = scipy.interpolate.interpolate.interp1d(time, dx, axis=0, fill_value='extrapolate')
    # solving linear system for all timesteps
    F_ = np.array([np.dot(dx_[5:].T, np.linalg.inv(dX[5:].T)) for dx_ in dx])
    F_interp = scipy.interpolate.interp1d(time, F_, axis=0, fill_value='extrapolate')
    return F_interp, dX

def area(F, dX):
    dx=[np.dot(dX_inst, F) for dX_inst in dX]
    A_temp=[]
    for inds in [[7, 0, 4, 3],[5, 0, 4, 1],[2,0,3,1]]:
        A_temp.append(np.linalg.norm(np.cross(dx[inds[0]] - dx[inds[1]], dx[inds[2]] - dx[inds[3]])) / 2)
    A = np.array([A_temp]*3)
    return A

def traction_bndout(bndout, node_order=None):
    bndout_list=list(bndout.values())
    if node_order != None:
        bndout_list = [bndout[ind] for ind in node_order]
    face_ind_order=[[[1,2,5,6], [0, 3, 4, 7]],
               [[2,3,6,7], [0, 1, 4, 5]],
               [[4, 5, 6, 7], [0, 1, 2, 3]]]
    R = list(range(9))
    for i in range(3):
        for j, jinds in enumerate(face_ind_order):
            pos_sum =np.sum([bndout_list[ind].F[:,i] for ind in jinds[1]], axis=0)
            neg_sum =np.sum([bndout_list[ind].F[:,i] for ind in jinds[0]], axis=0)
            R[i * 3 + j] = (pos_sum-neg_sum) / 2.

    R=np.array(R).reshape(3,3,-1).swapaxes(0,2)
    traction = -1 * R
    return traction

def traction_rcforc(rcforc, ftorder=None):
        '''order: X0, X1, Y0, Y1, Z0, Z1
        NOT TESTED'''
        rcforc_list=list(rcforc.values())
        if ftorder != None:
            rcforc_list=[rcforc[ind] for ind in ftorder]
        R=list(range(9))
        for i in range(3):
            for j, jinds in enumerate([[1, 0],[3, 2],[5, 4]]):
                R[i*3+j] = (rcforc_list[jinds[0]].R[:,i] - rcforc_list[jinds[1]].R[:,i]) / 2.

        R = np.array(R).reshape(3, 3, -1).swapaxes(0, 2)
        traction = -1 * R
        return traction
        #Rxx = (rcforc[1].Rx - rcforc[0].Rx) / 2.
        #Rxy = (rcforc[1].Ry - rcforc[0].Ry) / 2.
        #Rxz = (rcforc[1].Rz - rcforc[0].Rz) / 2.

        #Ryx = (rcforc[3].Rx - rcforc[2].Rx) / 2.
        #Ryy = (rcforc[3].Ry - rcforc[2].Ry) / 2.
        #Ryz = (rcforc[3].Rz - rcforc[2].Rz) / 2.

        #Rzx = (rcforc[5].Rx - rcforc[4].Rx) / 2.
        #Rzy = (rcforc[5].Ry - rcforc[4].Ry) / 2.
        #Rzz = (rcforc[5].Rz - rcforc[4].Rz) / 2.

        #forceP=-1*np.array([[Rxx, Rxy, Rxz],
         #                   [Ryx, Ryy, Ryz],
          #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        #df= P N ds
        #forceP = -1 * np.array([[Rxx, Ryx, Rzx],
                                #[Rxy, Ryy, Rzy],
                                #[Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)



def gasContr(self, relativeDensity):
    #P1V1/T1= P2V2/T2
    V1=1-relativeDensity
    V2=np.linalg.det(self.F)-relativeDensity
    P1=0.1#Mpa
    P2=P1*V1/V2
    P2gauge=P2-P1
    self.gasP=np.array([np.identity(3)*P2gaugeVal for P2gaugeVal in P2gauge])

def engStress(self, relativeDensity, source='bndout'):
    self.stressP(source=source)
    self.gasContr(relativeDensity=relativeDensity)
    return self.P-self.gasP

def engStrain(self):
    return self.F-np.array([np.identity(3)]*len(self.F))
