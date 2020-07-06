import os
from collections import namedtuple
import numpy as np
import scipy.interpolate

class SimResults(object):
    def __init__(self, sim_folder, periodic=False):
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
        self.periodic=periodic
    def PK1(self, source='bndout'):
        if self.periodic == True:
            if source == 'bndout':
                time = list(self.bndout.values())[0].time
                traction = traction_bndout_periodic(self.bndout)

            elif source == 'nodfor':
                time = list(self.nodfor.values())[0].time
                traction = traction_nodfor_periodic(self.nodfor)

        else:
            if source == 'bndout':
                time=list(self.bndout.values())[0].time
                traction = traction_bndout(self.bndout, node_order_element(self.nodout))

            elif source == 'rcforc':
                time = list(self.rcforc.values())[0].time
                traction = traction_rcforc(self.rcforc, node_order_element(self.nodout))


        PK1 = traction / self.A0()
        PK1_interp = scipy.interpolate.interp1d(time, PK1, axis=0, fill_value='extrapolate')
        return PK1_interp(self.time)


    def PK1_plate(self, source='nodfor', plate='z'):
        if source == 'nodfor':
            time = list(self.nodfor.values())[0].time
            traction = traction_nodfor(self.nodfor)
        dir_dict={'x':0, 'y':1, 'z':2}
        PK1 = traction / self.A0()
        PK1_interp = scipy.interpolate.interp1d(time, PK1, axis=0, fill_value='extrapolate')
        return PK1_interp(self.time)[:,dir_dict[plate],:]

    def F(self):
        if self.periodic == True:
            F_interp, _ = def_gradient_periodic(self.nodout, dX_ref=self.dX_ref)
        else:
            F_interp, _ = def_gradient(self.nodout, dX_ref=self.dX_ref)
        return F_interp(self.time)

    def inf_strain(self):
        F_temp=self.F()
        return 0.5*(np.transpose(F_temp, (0, 2, 1))+F_temp)-np.identity(3)

    def inf_strain_plate(self):
        F_temp=self.F()
        return 0.5*(np.transpose(F_temp, (0, 2, 1))+F_temp)-np.identity(3)

    def A0(self):
        if self.periodic == True:
            F_interp, dX = def_gradient_periodic(self.nodout, dX_ref=self.dX_ref)
            A0 = area_periodic(F_interp(0.0), dX)
        else:
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
    bnd_tup = namedtuple('bndTup', ['time', 'bndsid', 'F'])
    with open(r'{}\bndout'.format(sim_folder)) as bndout:
        lines = bndout.readlines()

    output_time_data = np.array(
        [[i, float(line.split()[-1])] for i, line in enumerate(lines) if r' t= ' in line])
    if len(output_time_data) == 0:
        return []

    timestamps = output_time_data[:, 1]
    output_time_ind = list(map(int, output_time_data[:, 0]))
    bndsids = []
    for line in lines[:output_time_ind[0]]:
        if 'Boundary disp' in line:
            line = line.split()
            bndsids.append(line[0])

    numnodes = (output_time_ind[1] - output_time_ind[0]) - 6
    header = 4
    node_values = [[] for _ in range(numnodes)]
    for ind, time in zip(output_time_ind, timestamps):
        for i in range(numnodes):
            line = lines[ind + header + i]
            line=line.split()
            nid = int(line[1])
            sid = int(line[12])
            values = [nid] + [sid] + [float(force) for force in line[3:8:2]]
            node_values[i].append(values)
    bndout_dict = {}
    #for bndsid in bndsids:
    for j in range(numnodes):
        values = np.array(node_values[j])
        bndout_dict[int(values[0, 0])] = bnd_tup(timestamps, int(values[0, 1]), values[:, 2:5])
    return bndout_dict

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
        [[int(i), float(line.split()[-2])] for i, line in enumerate(lines) if 'at time' in line])
    if len(output_time_data) == 0:
        return []

    numnodes_scale = 1
    if output_time_data[0][1] == output_time_data[1][1]:
        output_time_data = output_time_data[::2, :]
        numnodes_scale=2
    timestamps = output_time_data[:, 1]
    output_time_ind = output_time_data[:, 0]
    numnodes = int((output_time_ind[1] - output_time_ind[0]) / numnodes_scale - 6)
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
    #output_time_ind.append(-)
    for ind, time in enumerate(timestamps):
        if ind == len(timestamps)-1:
            lines_temp = lines[output_time_ind[ind]:]
        else:
            lines_temp=lines[output_time_ind[ind]:output_time_ind[ind+1]]
        for line in lines_temp:
            if 'nodal group output number' in line:
                group_id = int(line.split()[-1])
            if 'xtotal' in line:
                line = line.replace('\n', '')
                line = line.split()
                if ind==0:
                    force_values[group_id] = []
                force_values[group_id].append([float(value) for value in line[1:-1:2]])
    nodfor = {}
    for node_group in force_values.items():
        nodfor[group_to_setid[node_group[0]]] = nodfor_tup(timestamps, np.array(node_group[1]))
    return nodfor

def node_order_element(nodout):
    '''Z-axis point up, second node along x-axis'''
    corner_nodes = list(range(8))
    box_array = np.array([node.A[0] for node in nodout.values()])
    box_array_x = [min(box_array[:,0]), max(box_array[:,0])]
    box_array_y = [min(box_array[:, 1]), max(box_array[:, 1])]
    box_array_z = [min(box_array[:, 2]), max(box_array[:, 2])]
    loc_bools =         [[box_array_x[0], box_array_y[0], box_array_z[0]],
                         [box_array_x[1], box_array_y[0], box_array_z[0]],
                         [box_array_x[1], box_array_y[1], box_array_z[0]],
                         [box_array_x[0], box_array_y[1], box_array_z[0]],
                         [box_array_x[0], box_array_y[0], box_array_z[1]],
                         [box_array_x[1], box_array_y[0], box_array_z[1]],
                         [box_array_x[1], box_array_y[1], box_array_z[1]],
                         [box_array_x[0], box_array_y[1], box_array_z[1]]]


    for node in nodout.items():
        for ind, loc_bool in enumerate(loc_bools):
            if all((node[1].A[0] == loc_bool)):
                corner_nodes[ind] = node[0]
    return corner_nodes

def def_gradient(nodout, dX_ref=None):
    node_order=node_order_element(nodout)
    dX = np.array([nodout[node].A[0] for node in node_order])
    dX_org = dX
    if type(dX_ref) == type(np.array([])):
        loc_bools = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1,1, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 1,1]])
        dX=loc_bools*dX_ref
    # deformed vectors
    dx = np.array([nodout[node].A for node in node_order]).swapaxes(0, 1)
    time = nodout[node_order[0]].time
    #dx_interp = scipy.interpolate.interpolate.interp1d(time, dx, axis=0, fill_value='extrapolate')
    # solving linear system for all timesteps
    F_ = np.array([np.dot(dx_[5:].T, np.linalg.inv(dX_org[5:].T)) for dx_ in dx])
    F_interp = scipy.interpolate.interp1d(time, F_, axis=0, fill_value='extrapolate')
    return F_interp, dX

def def_gradient_periodic(nodout, dX_ref=None):
    node_order=list(nodout.keys())
    node_order.sort()
    dX = np.array([nodout[node].A[0] for node in node_order[1:]])
    dX_org = dX
    if type(dX_ref) == type(np.array([])):
        loc_bools = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1,1, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 1],
                     [1, 1, 1],
                     [0, 1,1]])
        dX=loc_bools*dX_ref
    # deformed vectors
    dx = np.array([nodout[node].A for node in node_order[1:]]).swapaxes(0, 1)
    time = nodout[node_order[0]].time
    #dx_interp = scipy.interpolate.interpolate.interp1d(time, dx, axis=0, fill_value='extrapolate')
    # solving linear system for all timesteps
    F_ = np.array([np.dot(dx_.T, np.linalg.inv(dX_org.T)) for dx_ in dx])
    F_interp = scipy.interpolate.interp1d(time, F_, axis=0, fill_value='extrapolate')
    return F_interp, dX

def area(F_, dX):
    dx=[np.dot(dX_inst, F_) for dX_inst in dX]
    A_temp=[]
    for inds in [[7, 0, 4, 3],[5, 0, 4, 1],[2,0,3,1]]:
        A_temp.append(np.linalg.norm(np.cross(dx[inds[0]] - dx[inds[1]], dx[inds[2]] - dx[inds[3]])) / 2)
    A = np.array([A_temp]*3)
    return A

def area_periodic(F_, dX):
    dx=np.array([np.dot(dX_inst, F_) for dX_inst in dX])
    A=np.array([[dx[2,2]*dx[1,1]]*3,
                [dx[0, 0] * dx[2, 2]] * 3,
                [dx[0, 0] * dx[1, 1]] * 3])
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

def traction_bndout_periodic(bndout):
    bndout_list = list(bndout.values())
    Rxx = bndout_list[0].F[:, 0]
    Rxy = bndout_list[0].F[:, 1]
    Rxz = bndout_list[0].F[:, 2]

    Ryx = bndout_list[1].F[:, 0]
    Ryy = bndout_list[1].F[:, 1]
    Ryz = bndout_list[1].F[:, 2]

    Rzx = bndout_list[2].F[:, 0]
    Rzy = bndout_list[2].F[:, 1]
    Rzz = bndout_list[2].F[:, 2]

    # forceP=-1*np.array([[Rxx, Rxy, Rxz],
    #                   [Ryx, Ryy, Ryz],
    #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

    # df= P N ds
    R = np.array([[Rxx, Ryx, Rzx],
                  [Rxy, Ryy, Rzy],
                  [Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)
    traction = 1 * R
    return traction

def traction_nodfor(nodfor, node_order=None):
    nodfor_list=list(nodfor.values())
    if node_order != None:
        nodfor_list = [nodfor[ind] for ind in node_order]

    face_ind_order=[[[1,2,5,6], [0, 3, 4, 7]],
               [[2,3,6,7], [0, 1, 4, 5]],
               [[4, 5, 6, 7], [0, 1, 2, 3]]]
    R = list(range(9))
    for i in range(3):
        for j, jinds in enumerate(face_ind_order):
            pos_sum =np.sum([nodfor_list[ind].R[:,i] for ind in jinds[1]], axis=0)
            neg_sum =np.sum([nodfor_list[ind].R[:,i] for ind in jinds[0]], axis=0)
            R[i * 3 + j] = (pos_sum-neg_sum) / 2.

    R=np.array(R).reshape(3,3,-1).swapaxes(0,2)
    traction = -1 * R
    return traction

def stressP(self, source='bndout'):
    #bndDict, nodeOrder = self.readbndout()
    nodeDict = self.readnodout()
    nodeOrder = list(nodeDict.keys())
    nodeOrder.sort()
    dX = np.array([[nodeDict[node].x[0], nodeDict[node].y[0], nodeDict[node].z[0]] for node in nodeOrder[1:]])
    # deformed vectors
    dx = np.array([[nodeDict[node].x, nodeDict[node].y, nodeDict[node].z] for node in nodeOrder[1:]]).swapaxes(0,1).swapaxes(0, 2)
    dxtime = nodeDict[nodeOrder[0]].time
    #dxInterp = interp1d(dxtime, dx, axis=0, fill_value='extrapolate')

    # solving linear system for all timesteps
    Ftime = nodeDict[nodeOrder[0]].time
    Forg = np.array([np.dot(dxTemp.T, np.linalg.inv(dX.T)) for dxTemp in dx])
    Finterp = interp1d(Ftime, Forg, axis=0, fill_value='extrapolate')
    #
    # overallTime = np.insert(bndDict[nodeOrder[0]].time, 0, 0.0)
    # detF = np.linalg.det(F)
    if source == 'nodfor':
        nodfor = self.readnodfor()
        F = Finterp(nodfor[0].time)
        self.F = F

        Ax = np.linalg.norm(np.cross(dX[1], dX[2]))
        Ay = np.linalg.norm(np.cross(dX[0], dX[2]))
        Az = np.linalg.norm(np.cross(dX[0], dX[1]))
        A0 = np.array([[Ax, Ay, Az],
                       [Ax, Ay, Az],
                       [Ax, Ay, Az]])

        Rxx = nodfor[1].Rx
        Rxy = nodfor[1].Ry
        Rxz = nodfor[1].Rz

        Ryx = nodfor[2].Rx
        Ryy = nodfor[2].Ry
        Ryz = nodfor[2].Rz

        Rzx = nodfor[3].Rx
        Rzy = nodfor[3].Ry
        Rzz = nodfor[3].Rz

        # forceP=-1*np.array([[Rxx, Rxy, Rxz],
        #                   [Ryx, Ryy, Ryz],
        #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        # df= P N ds
        forceP = -1 * np.array([[Rxx, Ryx, Rzx],
                                [Rxy, Ryy, Rzy],
                                [Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        P = forceP / A0
        self.A0 = A0
        self.forceP = forceP
        self.P = P

    elif source == 'bndout':
        bndDict, nodeOrderBnd = self.readbndout()
        F = Finterp(bndDict[nodeOrderBnd[0]].time)
        self.F = F

        Ax = np.linalg.norm(np.cross(dX[1], dX[2]))
        Ay = np.linalg.norm(np.cross(dX[0], dX[2]))
        Az = np.linalg.norm(np.cross(dX[0], dX[1]))
        A0 = np.array([[Ax, Ay, Az],
                       [Ax, Ay, Az],
                       [Ax, Ay, Az]])

        Rxx = bndDict[nodeOrderBnd[0]].Fx
        Rxy = bndDict[nodeOrderBnd[0]].Fy
        Rxz = bndDict[nodeOrderBnd[0]].Fz

        Ryx = bndDict[nodeOrderBnd[1]].Fx
        Ryy = bndDict[nodeOrderBnd[1]].Fy
        Ryz = bndDict[nodeOrderBnd[1]].Fz

        Rzx = bndDict[nodeOrderBnd[2]].Fx
        Rzy = bndDict[nodeOrderBnd[2]].Fy
        Rzz = bndDict[nodeOrderBnd[2]].Fz

        # forceP=-1*np.array([[Rxx, Rxy, Rxz],
        #                   [Ryx, Ryy, Ryz],
        #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        # df= P N ds
        forceP = 1 * np.array([[Rxx, Ryx, Rzx],
                               [Rxy, Ryy, Rzy],
                               [Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

        P = forceP / A0
        self.forceP = forceP
        self.P = P

    else:
        raise Exception('{} was not recognised as source'.format(source))

def traction_nodfor_periodic(nodfor):
    nodfor_list=list(nodfor.values())
    Rxx = nodfor_list[1].R[:,0]
    Rxy = nodfor_list[1].R[:,1]
    Rxz = nodfor_list[1].R[:,2]

    Ryx = nodfor_list[2].R[:,0]
    Ryy = nodfor_list[2].R[:,1]
    Ryz = nodfor_list[2].R[:,2]

    Rzx = nodfor_list[3].R[:,0]
    Rzy = nodfor_list[3].R[:,1]
    Rzz = nodfor_list[3].R[:,2]

    # forceP=-1*np.array([[Rxx, Rxy, Rxz],
    #                   [Ryx, Ryy, Ryz],
    #                  [Rzx, Rzy, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)

    # df= P N ds
    R = np.array([[Rxx, Ryx, Rzx],
                            [Rxy, Ryy, Rzy],
                            [Rxz, Ryz, Rzz]]).swapaxes(0, 1).swapaxes(0, 2)
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
