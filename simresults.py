import os
from collections import namedtuple
import numpy as np
import scipy as sp

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
    bnd_dict = {}
    for j in range(numnodes):
        values = np.array(node_values[j])
        bnd_dict[int(values[0, 0])] = bnd_tup(timestamps, values[:, 1:4])
    return bnd_dict#, node_order

def readglstat(sim_folder):
    '''Reads timestamps and resultant forces on bnd file
    returns [t],[x,y,z]'''
    glstat_tup = namedtuple('glstatTup', ['time', 'time_step', 'kinetic_energy', 'internal_energy',
                                         'external_work', 'total_energy', 'energy_ratio', 'mass_increase_percent'])
    with open(r'{}\glstat'.format(sim_folder)) as glstat:
        lines = glstat.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if 'time......' in line])
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
    node_dict = {}
    for i in range(numnodes):
        values = np.array(node_values[i])
        node_dict[int(values[0, 0])] = nod_tup(timestamps, values[:, 1:4], values[:, 4:7], values[:, 10:13])
    return node_dict

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
    corner_nodes = [0] * 8
    for node in nodout_dict.values():
        if node.x[0] == 0.0 and node.y[0] == 0.0 and node.z[0] == 0.0:
            corner_nodes[0] = node.nid
        elif node.x[0] != 0.0 and node.y[0] == 0.0 and node.z[0] == 0.0:
            corner_nodes[1] = node.nid
        elif node.x[0] != 0.0 and node.y[0] != 0.0 and node.z[0] == 0.0:
            corner_nodes[2] = node.nid
        elif node.x[0] == 0.0 and node.y[0] != 0.0 and node.z[0] == 0.0:
            corner_nodes[3] = node.nid
        elif node.x[0] == 0.0 and node.y[0] == 0.0 and node.z[0] != 0.0:
            corner_nodes[4] = node.nid
        elif node.x[0] != 0.0 and node.y[0] == 0.0 and node.z[0] != 0.0:
            corner_nodes[5] = node.nid
        elif node.x[0] != 0.0 and node.y[0] != 0.0 and node.z[0] != 0.0:
            corner_nodes[6] = node.nid
        elif node.x[0] == 0.0 and node.y[0] != 0.0 and node.z[0] != 0.0:
            corner_nodes[7] = node.nid
    return corner_nodes

def def_gradient(nodout_dict):
    node_order=node_order_cube(nodout_dict)
    dX = np.array([[nodout_dict[node].x[0], nodout_dict[node].y[0], nodout_dict[node].z[0]] for node in node_order])
    # deformed vectors
    dx = np.array([[nodout_dict[node].x, nodout_dict[node].y, nodout_dict[node].z] for node in node_order]).swapaxes(0,
                                                                                                           1).swapaxes(
        0, 2)
    dxtime = nodout_dict[nodeOrder[0]].time
    dxInterp = np.interp1d(dxtime, dx, axis=0, fill_value='extrapolate')

    # solving linear system for all timesteps
    Ftime = nodout_dict[nodeOrder[0]].time
    Forg = np.array([np.dot(dxTemp[5:].T, np.linalg.inv(dX[5:].T)) for dxTemp in dx])
    Finterp = sp.interpolate.interp1d(Ftime, Forg, axis=0, fill_value='extrapolate')

def stressP(self, source='bndout'):
    bndDict, nodeOrderOut = self.readbndout()
    nodeDict = self.readnodout()
    nodeOrder=self.findNodeOrder(nodeOrderOut,nodeDict)
    dX = np.array([[nodeDict[node].x[0], nodeDict[node].y[0], nodeDict[node].z[0]] for node in nodeOrder])
    # deformed vectors
    dx = np.array([[nodeDict[node].x, nodeDict[node].y, nodeDict[node].z] for node in nodeOrder]).swapaxes(0,1).swapaxes(0, 2)
    dxtime = nodeDict[nodeOrder[0]].time
    dxInterp = interp1d(dxtime, dx, axis=0, fill_value='extrapolate')

    # solving linear system for all timesteps
    Ftime = nodeDict[nodeOrder[0]].time
    Forg = np.array([np.dot(dxTemp[5:].T, np.linalg.inv(dX[5:].T)) for dxTemp in dx])
    Finterp = interp1d(Ftime, Forg, axis=0, fill_value='extrapolate')



    #
    #overallTime = np.insert(bndDict[nodeOrder[0]].time, 0, 0.0)
    #detF = np.linalg.det(F)
    if source == 'rcforc':
        rcforc=self.readrcforc()
        F = Finterp(rcforc[0].time)
        self.F=F

        Ax = np.linalg.norm(np.cross(dX[7] - dX[0], dX[4] - dX[3]))/2
        Ay = np.linalg.norm(np.cross(dX[5] - dX[0], dX[4] - dX[1]))/2
        Az = np.linalg.norm(np.cross(dX[2] - dX[0], dX[3] - dX[1]))/2
        A0 = np.array([[Ax, Ay, Az],
                       [Ax, Ay, Az],
                       [Ax, Ay, Az]])

        Rxx = (rcforc[1].Rx - rcforc[0].Rx) / 2.
        Rxy = (rcforc[1].Ry - rcforc[0].Ry) / 2.
        Rxz = (rcforc[1].Rz - rcforc[0].Rz) / 2.

        Ryx = (rcforc[3].Rx - rcforc[2].Rx) / 2.
        Ryy = (rcforc[3].Ry - rcforc[2].Ry) / 2.
        Ryz = (rcforc[3].Rz - rcforc[2].Rz) / 2.

        Rzx = (rcforc[5].Rx - rcforc[4].Rx) / 2.
        Rzy = (rcforc[5].Ry - rcforc[4].Ry) / 2.
        Rzz = (rcforc[5].Rz - rcforc[4].Rz) / 2.

        #forceP=-1*np.array([[Rxx, Rxy, Rxz],
         #                   [Ryx, Ryy, Ryz],
          #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        #df= P N ds
        forceP = -1 * np.array([[Rxx, Ryx, Rzx],
                                [Rxy, Ryy, Rzy],
                                [Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        self.A0=A0
        P=forceP/A0
        self.P=P

    elif source =='bndout':
        F = Finterp(bndDict[nodeOrder[0]].time)
        self.F=F
        PXX = (bndDict[nodeOrder[1]].Fx + bndDict[nodeOrder[2]].Fx + bndDict[nodeOrder[5]].Fx +
                             bndDict[nodeOrder[6]].Fx)
        PXX0 = (bndDict[nodeOrder[0]].Fx + bndDict[nodeOrder[3]].Fx + bndDict[nodeOrder[4]].Fx +
                              bndDict[nodeOrder[7]].Fx)
        PYY = (bndDict[nodeOrder[2]].Fy + bndDict[nodeOrder[3]].Fy + bndDict[nodeOrder[6]].Fy +
                             bndDict[nodeOrder[7]].Fy)
        PYY0 = (bndDict[nodeOrder[0]].Fy + bndDict[nodeOrder[1]].Fy + bndDict[nodeOrder[4]].Fy +
                              bndDict[nodeOrder[5]].Fy)
        PZZ = (bndDict[nodeOrder[4]].Fz + bndDict[nodeOrder[5]].Fz + bndDict[nodeOrder[6]].Fz +
                             bndDict[nodeOrder[7]].Fz)
        PZZ0 = (bndDict[nodeOrder[0]].Fz + bndDict[nodeOrder[1]].Fz + bndDict[nodeOrder[2]].Fz +
                              bndDict[nodeOrder[3]].Fz)
        PXY = (bndDict[nodeOrder[1]].Fy + bndDict[nodeOrder[2]].Fy + bndDict[nodeOrder[5]].Fy +
                             bndDict[nodeOrder[6]].Fy)
        PXY0 = (bndDict[nodeOrder[0]].Fy + bndDict[nodeOrder[3]].Fy + bndDict[nodeOrder[4]].Fy +
                              bndDict[nodeOrder[7]].Fy)
        PYX = (bndDict[nodeOrder[2]].Fx + bndDict[nodeOrder[3]].Fx + bndDict[nodeOrder[6]].Fx +
                             bndDict[nodeOrder[7]].Fx)
        PYX0 = (bndDict[nodeOrder[0]].Fx + bndDict[nodeOrder[1]].Fx + bndDict[nodeOrder[4]].Fx +
                              bndDict[nodeOrder[5]].Fx)
        PZX = (bndDict[nodeOrder[4]].Fx + bndDict[nodeOrder[5]].Fx + bndDict[nodeOrder[6]].Fx +
                             bndDict[nodeOrder[7]].Fx)
        PZX0 = (bndDict[nodeOrder[0]].Fx + bndDict[nodeOrder[1]].Fx + bndDict[nodeOrder[2]].Fx +
                              bndDict[nodeOrder[3]].Fx)
        PXZ = (bndDict[nodeOrder[1]].Fz + bndDict[nodeOrder[2]].Fz + bndDict[nodeOrder[5]].Fz +
                             bndDict[nodeOrder[6]].Fz)
        PXZ0 = (bndDict[nodeOrder[0]].Fz + bndDict[nodeOrder[3]].Fz + bndDict[nodeOrder[4]].Fz +
                              bndDict[nodeOrder[7]].Fz)
        PZY = (bndDict[nodeOrder[4]].Fy + bndDict[nodeOrder[5]].Fy + bndDict[nodeOrder[6]].Fy +
                             bndDict[nodeOrder[7]].Fy)
        PZY0 = (bndDict[nodeOrder[0]].Fy + bndDict[nodeOrder[1]].Fy + bndDict[nodeOrder[2]].Fy +
                              bndDict[nodeOrder[3]].Fy)
        PYZ = (bndDict[nodeOrder[2]].Fz + bndDict[nodeOrder[3]].Fz + bndDict[nodeOrder[6]].Fz +
                             bndDict[nodeOrder[7]].Fz)
        PYZ0 = (bndDict[nodeOrder[0]].Fz + bndDict[nodeOrder[1]].Fz + bndDict[nodeOrder[4]].Fz +
                              bndDict[nodeOrder[5]].Fz)

        #P = np.array([      [PXX, PYX, PXZ],
        #                    [PYX, PYY, PYZ],
        #                    [PZX, PZY, PZZ]]).swapaxes(0, 1).swapaxes(0, 2)
        P = np.array([[PXX, PYX, PZX],
                      [PXY, PYY, PZY],
                      [PXZ, PYZ, PZZ]]).swapaxes(0, 1).swapaxes(0, 2)

        #P0 = np.array([[-PXX0, -PYX0, -PZX0],
        #                     [-PXY0, -PYY0, -PZY0],
        #                     [-PXZ0, -PYZ0, -PZZ0]]).swapaxes(0, 1).swapaxes(0, 2)
        Ax = np.linalg.norm(np.cross(dX[7] - dX[0], dX[4] - dX[3])) / 2
        Ay = np.linalg.norm(np.cross(dX[5] - dX[0], dX[4] - dX[1])) / 2
        Az = np.linalg.norm(np.cross(dX[2] - dX[0], dX[3] - dX[1])) / 2
        A0 = np.array([[Ax, Ay, Az],
                       [Ax, Ay, Az],
                       [Ax, Ay, Az]])

        self.P = P/A0
        #return F, P


    else:
        raise Exception('{} was not recognised as source'.format(source))

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
