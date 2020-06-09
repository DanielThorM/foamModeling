#sys.path.insert(0, r'C:\Users\danieltm\OneDrive - NTNU\Python_Github\keywordGenerator')
#sys.path.insert(0, r'C:\Users\danieltm\OneDrive - NTNU\Python_Github\tessToPy')
import importlib as imp
#import tessellations as ts
#import meshmodel as mm
#import keywordGenerator as kw
#imp.reload(mm)
#imp.reload(kw)
import numpy as np
import os
import subprocess


class BoundaryConditions:#
    def __init__(self, keyword, mesh_geometry):
        self.keyword=keyword
        self.mesh_geometry = mesh_geometry
    def non_periodic(self, def_gradient, rot_dof=0, phi=0.0, soft=1, surf_contact=True):
        self.keyword.comment_block('Boundary conditions')
        side_parts = [surf[0] for surf in self.mesh_geometry.find_side_surfs() if surf != []]
        side_elements = [self.mesh_geometry.shell_elements[elem].id_ for surf in side_parts for elem in self.mesh_geometry.surfs[surf].elem_ids]
        plane_locs= [0.0, self.mesh_geometry.domain_size[0],
                        0.0, self.mesh_geometry.domain_size[1],
                        0.0, self.mesh_geometry.domain_size[2]]
        planes = ['x', 'x', 'y', 'y', 'z', 'z']
        face_nodes_list = []
        i=0
        for plane, plane_loc in zip(planes, plane_locs):
            nodes = list(set(self.mesh_geometry.create_node_list_in_plane(plane=plane, plane_loc=plane_loc))
                          - set([node.id_ for node in self.mesh_geometry.plate_corner_nodes])) #remove plate corner nodes
            face_nodes_list.append(nodes)
            i += 1
        face_core_nodes_list = []
        for i, plane in enumerate(['x', 'y', 'z']):
            if plane == 'x':
                rem_nodes = set([node for node_list in face_nodes_list[2:] for node in node_list])
            elif plane == 'y':
                rem_nodes = set([node for node_list in [*face_nodes_list[:2], *face_nodes_list[4:]] for node in node_list])
            elif plane == 'z':
                rem_nodes = set([node for node_list in face_nodes_list[:4] for node in node_list])
            face_core_nodes_list.append(list(set(face_nodes_list[i * 2]) - rem_nodes))
            face_core_nodes_list.append(list(set(face_nodes_list[(i * 2) + 1]) - rem_nodes))

        full_side_part_list = self.mesh_geometry.find_side_surfs()
        side_node_lists = [[], [], [], [], [], []]
        if len(side_parts) == 2:
            plane_combinations={0:[[[1], [0, 0, 0], [0]],
                                   [[1], [0, 0, 0], [0]],
                                   [[0], [1, 0, 1], [0]],
                                   [[0], [1, 0, 1], [0]],
                                   [[0], [0, 0, 0], [1]],
                                   [[0], [0, 0, 0], [1]]],
                                2:[[[0], [0, 0, 0], [1]],
                                   [[0], [0, 0, 0], [1]],
                                   [[1], [0, 0, 0], [0]],
                                   [[1], [0, 0, 0], [0]],
                                   [[0], [1, 2, 3], [0]],
                                   [[0], [1, 2, 3], [0]]],
                                4:[[[0], [1, 4, 5], [0]],
                                   [[0], [1, 4, 5], [0]],
                                   [[0], [0, 0, 0], [1]],
                                   [[0], [0, 0, 0], [1]],
                                   [[1], [0, 0, 0], [0]],
                                   [[1], [0, 0, 0], [0]]]}
            for i in range(0, 6, 2):
                if full_side_part_list[i] != []:
                    plane_combination = plane_combinations[i]
                    break
        elif len(side_parts) == 4:
            plane_combinations = {4: [[[1], [0, 0, 0], [0]],
                                      [[1], [0, 0, 0], [0]],
                                      [[0], [1, 0, 1], [0]],
                                      [[0], [1, 0, 1], [0]],
                                      [[0], [0, 0, 0], [1]],
                                      [[0], [0, 0, 0], [1]]],
                                  0: [[[0], [0, 0, 0], [1]],
                                      [[0], [0, 0, 0], [1]],
                                      [[1], [0, 0, 0], [0]],
                                      [[1], [0, 0, 0], [0]],
                                      [[0], [1, 2, 3], [0]],
                                      [[0], [1, 2, 3], [0]]],
                                  2: [[[0], [1, 4, 5], [0]],
                                      [[0], [1, 4, 5], [0]],
                                      [[0], [0, 0, 0], [1]],
                                      [[0], [0, 0, 0], [1]],
                                      [[1], [0, 0, 0], [0]],
                                      [[1], [0, 0, 0], [0]]]}
            for i in range(0, 6, 2):
                if full_side_part_list[i] == []:
                    plane_combination = plane_combinations[i]
                    break
        else:
            plane_combination =     [[[1], [0, 0, 0], [0]],
                                      [[1], [0, 0, 0], [0]],
                                      [[0], [1, 0, 1], [0]],
                                      [[0], [1, 0, 1], [0]],
                                      [[0], [0, 0, 0], [1]],
                                      [[0], [0, 0, 0], [1]]]

        for i in range(6):
            side_node_lists[i].extend(face_nodes_list[i] * plane_combination[i][0][0])
            side_node_lists[i].extend(list(set(face_nodes_list[i])
                                           - set(face_nodes_list[plane_combination[i][1][1]])
                                           - set(face_nodes_list[plane_combination[i][1][2]])
                                           )* plane_combination[i][1][0])
            side_node_lists[i].extend(face_core_nodes_list[i] * plane_combination[i][2][0])


        def tiebreak_contact_node_to_surface(self, soft):
            for i, side_part in enumerate(full_side_part_list):
                if side_part != [] and side_node_lists[i] != []:
                    side_part=side_part[0]
                    self.keyword.set_node_list(nsid=(i + 101), node_list=side_node_lists[i])
                    if surf_contact == True:
                        node_list = []
                        for face in side_faces_list[i]:
                            for element in self.mesh_geometry.surfs[face].elem_ids:
                                node_list.extend(self.mesh_geometry.shell_elements[element].node_ids)
                        side_faces_nodes = list(set(node_list) - set(side_node_lists[i]))
                        self.keyword.set_node_list(nsid=(i + 301), node_list=side_faces_nodes)
                        self.keyword.contact_automatic_nodes_to_surface(cid=(i + 301), ssid=(i + 301), msid=side_part,
                                                                        soft=soft, ignore=1,
                                                                        depth=1)
                    self.keyword.contact_tiebreak_nodes_to_surface(cid=(i + 101), ssid=(i + 101), msid=side_part, soft=soft, ignore=1)



        if phi == 1.0:
            side_node_lists = self.mesh_geometry.find_beam_node_on_side()
            side_faces_list = [[]] * 6
            tiebreak_contact_node_to_surface(self, soft)
        else:
            if surf_contact==True:
                side_faces_list = self.mesh_geometry.find_parts_for_box_contact()
            tiebreak_contact_node_to_surface(self, soft)

        if rot_dof == 1:
            for i, coord_nodes in enumerate(self.mesh_geometry.coord_systems):
                if side_elements[i] != []:
                    self.keyword.define_coordinate_nodes(cid=(i + 1), nodes=coord_nodes, flag=1)
                    self.keyword.set_node_list(nsid=501 + i,
                                               node_list=face_core_nodes_list[i])
                    self.keyword.boundary_spc_set(bspcid=(i + 501), nsid=(i + 501), dofx=0, dofy=0, dofz=0, dofrx=1, dofry=1,
                                                dofrz=0, cid=(i + 1))
        corner_nodes = self.mesh_geometry.corner_nodes
        self.def_grad_prescription(def_gradient, corner_nodes, ref_node_set= [[nid.id_] for nid in self.mesh_geometry.plate_corner_nodes])
        self.keyword.database_hist_node(nids=[node.id_ for node in self.mesh_geometry.plate_corner_nodes])
        return self.keyword

    def def_grad_prescription(self, def_gradient, ref_nodes, ref_node_set = None):
        disp_type = self.keyword.disp_type
        for i, node in enumerate(ref_nodes):
            coords = node.coord
            if ref_node_set != None:
                self.keyword.set_node_list(nsid=10 + 10 * i, node_list=ref_node_set[i])
            if len(def_gradient.shape) == 2:
                u_gradient = def_gradient - np.identity(3)
                u_disp = u_gradient.dot(coords)
                time_inc = self.keyword.endtim
                for j, node_disp in enumerate(u_disp):
                    # (self, lcid, dist, tstart, tend, triseFrac=0.1, v0=0.0)
                    if disp_type == 'disp':
                        vad=2
                        self.keyword.define_curve(lcid=10 * (i + 1) + (j + 1), abscissas=[0.0, time_inc],
                                                  ordinates=[0, abs(node_disp)])
                    elif disp_type == 'vel':
                        vad=0
                        self.keyword.define_curve_smooth(lcid=10 * (i + 1) + (j + 1), dist=abs(node_disp), tstart=0.0,
                                                         tend=self.keyword.endtim, trise_frac=0.1)

                    if ref_node_set != None:
                        self.keyword.boundary_prescribed_motion_set(bmsid=10 * (i + 1) + (j + 1), nsid=10 + 10 * i,
                                                                dof=(j + 1),
                                                                vad=vad, lcid=10 * (i + 1) + (j + 1), #if implicit: vad = 2, else vad = 0
                                                                sf=np.sign(node_disp),
                                                                birth=0.0, death=self.keyword.endtim)
                    else:
                        self.keyword.boundary_prescribed_motion_node(bmsid=10 * (i + 1) + (j + 1), nid=node.id_, dof=(j + 1),
                                                                     vad=vad, lcid=10 * (i + 1) + (j + 1),
                                                                     sf=np.sign(node_disp),
                                                                     birth=0.0, death=self.keyword.endtim)




            else:
                u_gradient = np.array([temp - np.identity(3) for temp in def_gradient])
                u_disp = np.array([temp.dot(coords) for temp in u_gradient])
                du_disp = u_disp - np.insert(u_disp[:1, :], 0, np.array([0, 0, 0]), axis=0)
                time_inc = self.keyword.endtim / len(du_disp)
                for k, du_disp_step in enumerate(du_disp):
                    for j, node_disp in enumerate(du_disp_step):
                        np.sign(node_disp)
                        if disp_type == 'disp':
                            vad = 2
                            self.keyword.define_curve(lcid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                      abscissas=[0.0, time_inc],
                                                      ordinates=[0, abs(node_disp)])
                        elif disp_type == 'vel':
                            vad = 0
                            self.keyword.define_curve_smooth(lcid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                             dist=abs(node_disp),
                                                             tstart=0.0,
                                                             tend=time_inc, trise_frac=0.1)

                        if ref_node_set != None:
                            self.keyword.boundary_prescribed_motion_set(bmsid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                                        nsid=10 + 10 * i, dof=(j + 1),
                                                                        vad=vad,
                                                                        lcid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                                        sf=np.sign(node_disp), birth=k * time_inc,
                                                                        death=(k + 1) * time_inc)
                        else:

                            self.keyword.boundary_prescribed_motion_node(bmsid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                                         nid=node.id_, dof=(j + 1),
                                                                         vad=vad,
                                                                         lcid=1000 * (k + 1) + 10 * (i + 1) + (j + 1),
                                                                         sf=np.sign(node_disp), birth=k * time_inc,
                                                                         death=(k + 1) * time_inc)
    ##################################################################################3
    #Periodic
    ##################################################################################################################

    def get_nid_list(self, master_node, slave_nodes, corner_ref_nodes, origo_ref_nodes):
        slave_node_ids = [slave_node[0] for slave_node in slave_nodes]
        corner_ref_nodes = [node for cnode, ref_node in zip(corner_ref_nodes, origo_ref_nodes) for node in [cnode, ref_node]]
        return [master_node] + slave_node_ids + corner_ref_nodes #nidList=[master_node] + slave_nod_ids + corner_ref_nodes

    def get_coef_list(self, nid_list, slave_nodes, sorting='paired'):
        coef_list_list = []
        for slaveNode in slave_nodes:
            periodicity = slaveNode[1:]
            coef_list = [1]
            for nid in nid_list[1:-6]:
                if nid == slaveNode[0]:
                    coef_list.append(-1)
                else:
                    coef_list.append(0)
            for i, period in enumerate(periodicity):
                if period == 0:
                    coef_list.extend([0, 0])
                else:
                    coef_list.extend([1 * np.sign(period), -1 * np.sign(period)])
            coef_list_list.append(coef_list)

        coef_list_list = np.array(coef_list_list)
        if sorting == 'paired':
            for j in range(len(coef_list_list) - 1):
                for i, coef_list in enumerate(coef_list_list[j + 1:]):
                    coef_list_list[i + j + 1] = coef_list - coef_list_list[j]
            return coef_list_list
        elif sorting == 'identity':
            for j in range(len(coef_list_list) - 1):
                for i, coef_list in enumerate(coef_list_list[j + 1:]):
                    coef_list_list[i + j + 1] = coef_list - coef_list_list[j]
            for j in range(len(coef_list_list) - 1):
                coef_list_list[-(j + 2)] = coef_list_list[-(j + 2)] + coef_list_list[-(j + 1)]
            return coef_list_list
        elif sorting == 'none':
            return coef_list_list
        else:
            raise Exception('Invalid sorting key: {}. Choose "paired","identity" or "none" '.format(sorting))

    def match_nid_coef(self, nid_list, coeff_list, ref_node=None):
        temp_list = [[nid, coeff] for nid, coeff in zip(nid_list, coeff_list) if coeff != 0 and nid != ref_node]
        temp_nid_list = list(map(int, np.array(temp_list)[:,0]))
        temp_coeff_list = list(np.array(temp_list)[:,1])
        return temp_nid_list, temp_coeff_list

    def periodic_linear_local(self, def_gradient):
        constrained_id_counter=9000001
        ref_elements = list(self.mesh_geometry.solid_elements.values())
        ref_nodes = [ref_elem.node_ids[0] for ref_elem in ref_elements]
        self.keyword.boundary_spc_node(bspcid=9999, nid=ref_nodes[0], dofx=1, dofy=1, dofz=1)
        corner_ref_nodes = ref_nodes[1:]
        origo_ref_nodes = [ref_nodes[0]]*3
        used_nodes_list=[]
        self.keyword.comment_block('Periodic constraints')
        master_node_list = [node.id_ for node in self.mesh_geometry.nodes.values() if node.master_to != []]
        for master_node in master_node_list:
            temp_slave_node = self.mesh_geometry.nodes[master_node].master_to
            slave_nodes = [temp_slave_node[i*4:i*4+4] for i in range(len(temp_slave_node[::4]))]
            nid_list = self.get_nid_list(master_node, slave_nodes, corner_ref_nodes, origo_ref_nodes)
            for nid in nid_list[:-6]:
                if nid in used_nodes_list:
                    print(nid_list)
                    raise Exception('Node {} used twice in relation'.format(nid))
                else:
                    used_nodes_list.append(nid)
            used_nodes_list.append(master_node)
            coeff_list = self.get_coef_list(nid_list, slave_nodes, sorting='paired')
            for coeffs in coeff_list: #coeffs = coeff_list[0]
                temp_nids, temp_coeffs = self.match_nid_coef(nid_list, coeffs, ref_node=ref_nodes[0])
                for direction in range(0, 3):
                    self.keyword.constrained_linear_local(lcid=constrained_id_counter, nid_list=temp_nids,
                                                        coeff_list=temp_coeffs, direction=direction + 1)
                    constrained_id_counter += 1

        #####################################################################
        #Element displacemente
        #####################################################################
        self.def_grad_prescription(def_gradient, [self.mesh_geometry.nodes[node] for node in corner_ref_nodes])
        self.keyword.database_hist_node(nids=ref_nodes)
        return self.keyword

    def periodic_multiple_global(self, def_gradient):
        constrained_id_counter = 9000001
        ref_elements = list(self.mesh_geometry.solid_elements.values())
        ref_nodes = [ref_elem.node_ids[0] for ref_elem in ref_elements]
        corner_ref_nodes = ref_nodes[1:]
        origo_ref_nodes = [ref_nodes[0]] * 3
        used_nodes_list = []
        global_constr_list = [[], [], []]
        self.keyword.comment_block('Periodic constraints')
        master_node_list = [node.id_ for node in self.mesh_geometry.nodes.values() if node.master_to != []]
        for master_node in master_node_list:
            temp_slave_node = self.mesh_geometry.nodes[master_node].master_to
            slave_nodes = [temp_slave_node[i * 4:i * 4 + 4] for i in range(len(temp_slave_node[::4]))]
            nid_list = self.get_nid_list(master_node, slave_nodes, corner_ref_nodes, origo_ref_nodes)
            for nid in nid_list[:-6]:
                if nid in used_nodes_list:
                    print(nid_list)
                    raise Exception('Node {} used twice in relation'.format(nid))
                else:
                    used_nodes_list.append(nid)
            used_nodes_list.append(master_node)
            coeff_list = self.get_coef_list(nid_list, slave_nodes, sorting='identity')
            for coeffs in coeff_list:
                temp_nids, temp_coeffs = self.match_nid_coef(nid_list, coeffs, ref_node=ref_nodes[0])
                for direction in range(0, 3):
                    temp_global_dict = {}
                    temp_global_dict['nmp'] = len(temp_nids)
                    temp_global_dict['nid_list'] = temp_nids
                    temp_global_dict['coeff_list'] = temp_coeffs
                    global_constr_list[direction].append(temp_global_dict)

        for direction in range(0, 3):
            self.keyword.constrained_multiple_global(id=constrained_id_counter, constr_list=global_constr_list[direction],
                                                   direction=direction + 1)
            constrained_id_counter += 1

        #####################################################################
        # Element displacemente
        #####################################################################
        self.def_grad_prescription(def_gradient, [self.mesh_geometry.nodes[node] for node in ref_nodes],
                                   ref_node_set=[ref_elem.node_ids[1:] for ref_elem in ref_elements])
        self.keyword.database_hist_node(nids=ref_nodes)
        self.keyword.set_node_list(nsid=888888 - 1, node_list=[ref_elements[0].node_ids[0]])
        self.keyword.database_nodfor_group(nsid=888888 - 1)
        for i in range(3):
            self.keyword.set_node_list(nsid=888888 + i, node_list=[ref_elements[i+1].node_ids[0]])
            self.keyword.database_nodfor_group(nsid=888888 + i)

        return self.keyword

    def solid_element_single(self, def_gradient):
        corner_nodes = self.mesh_geometry.corner_nodes
        self.def_grad_prescription(def_gradient, corner_nodes)
        self.keyword.set_node_list(nsid=99, node_list=[node.id_ for node in corner_nodes])
        self.keyword.database_hist_node_set(nsids=[99])
        return self.keyword

    def solid_elements_enclosed(self, def_gradient, rot_dof=0, soft=1):
        self.keyword.comment_block('Boundary conditions')
        side_parts = [surf[0] for surf in self.mesh_geometry.find_side_surfs() if surf != []]
        side_elements = [self.mesh_geometry.shell_elements[elem].id_ for surf in side_parts for elem in self.mesh_geometry.surfs[surf].elem_ids]
        plane_locs= [0.0, self.mesh_geometry.domain_size[0],
                        0.0, self.mesh_geometry.domain_size[1],
                        0.0, self.mesh_geometry.domain_size[2]]
        planes = ['x', 'x', 'y', 'y', 'z', 'z']
        face_nodes_list = []
        i=0
        for plane, plane_loc in zip(planes, plane_locs):
            nodes = list(set(self.mesh_geometry.create_node_list_in_plane(plane=plane, plane_loc=plane_loc))
                          - set([node.id_ for node in self.mesh_geometry.plate_corner_nodes])) #remove plate corner nodes
            face_nodes_list.append(nodes)
            i += 1
        face_core_nodes_list = []
        for i, plane in enumerate(['x', 'y', 'z']):
            if plane == 'x':
                rem_nodes = set([node for node_list in face_nodes_list[2:] for node in node_list])
            elif plane == 'y':
                rem_nodes = set([node for node_list in [*face_nodes_list[:2], *face_nodes_list[4:]] for node in node_list])
            elif plane == 'z':
                rem_nodes = set([node for node_list in face_nodes_list[:4] for node in node_list])
            face_core_nodes_list.append(list(set(face_nodes_list[i * 2]) - rem_nodes))
            face_core_nodes_list.append(list(set(face_nodes_list[(i * 2) + 1]) - rem_nodes))



        side_node_lists = [face_nodes_list[0], face_nodes_list[1],
                         list(set(face_nodes_list[2]) - set(face_nodes_list[0]) - set(face_nodes_list[1])),
                         list(set(face_nodes_list[3]) - set(face_nodes_list[0]) - set(face_nodes_list[1])),
                         face_core_nodes_list[4],
                         face_core_nodes_list[5]] #Remove nodes which would have been duplicated at the edges #
        ## Specifiy compression platten direction to assign nodes to correct plane


        def tiebreak_contact_node_to_surface(self, soft):
            for i, side_part in enumerate(side_parts):
                self.keyword.set_node_list(nsid=(i + 101), node_list=side_node_lists[i])
                self.keyword.contact_tiebreak_nodes_to_surface(cid=(i + 101), ssid=(i + 101), msid=side_part, fs=0.0,
                                                           fd=0.0, soft=soft, ignore=1)
                self.keyword.contact_force_transducer(cid=(i + 201), ssid=side_part)

        tiebreak_contact_node_to_surface(self, soft)

        if rot_dof == 1:
            for i, coord_nodes in enumerate(self.mesh_geometry.coord_systems):
                if side_elements[i] != []:
                    self.keyword.define_coordinate_nodes(cid=(i + 1), nodes=coord_nodes, flag=1)
                    self.keyword.set_node_list(nsid=501 + i,
                                               node_list=face_core_nodes_list[i])
                    self.keyword.boundary_spc_set(bspcid=(i + 501), nsid=(i + 501), dofx=0, dofy=0, dofz=0, dofrx=1, dofry=1,
                                                dofrz=0, cid=(i + 1))
        corner_nodes = self.mesh_geometry.corner_nodes
        self.def_grad_prescription(def_gradient, corner_nodes, ref_node_set= [[nid.id_] for nid in self.mesh_geometry.plate_corner_nodes])
        self.keyword.database_hist_node(nids=self.mesh_geometry.plate_corner_nodes)
        return self.keyword

def periodic_template(tessellation, model_file_name, def_gradient, rho=0.05, phi=0.0, material_data={}, **kwargs):
    options = {
        'elem_type': 16,
        'strain_rate':  1.0,
        'size_coeff': 1.0,
        'strain_coeff': 1.0,
        'n_steps_coeff':500,
        'pert_nodes': 0.0,
        'pert_shell': 0.0,
        'tt_sigma':0.0,
        'csa_sigma': 0.0,
        'run': False,
        'return_copy': False,
        'sim_type':'implicit',
        'airbag':False,
        'beam_shape':'straight', #'marvi
        'beam_cs_shape':'round', #'tri'
        'shell_nip':7
    }
    options.update(kwargs)
    material = {
        'e':1500,
        'sigy':25.0,
        'etan':1.0,
        'pr':0.3,
        'ro':9.2e-10,
        'matfail':2.0,
        'soften':False,
        'stress':None,
        'strain':None,
        'rate_c':0.0,
        'rate_p':0.0,
        'rate_mod':None, #[slope1, slope2, base_rate, break_rate], eg. [0.075, 0.2, 1e-3, 3e1]
        'mat_type':'mat24', #'mat181'
        'fs':0.76,
        'fd':0.76,
        'dc':0.0
    }
    material.update(material_data)

    mesh_geometry = mm.FoamModel(tessellation) # mesh_geometry = LSDynaPerGeom(perTessGeometry, debug=True) #mesh_geometry = LSDynaPerGeom(tessellation, debug=True)
    keyword = kw.Keyword(model_file_name)# model_file_name = r'H:\thesis\periodic\representative\S05R1\ID1\testKey.key'
    keyword.comment_block('Control')
    keyword.control_structured()
    endtim = options['strain_coeff']*options['size_coeff'] / options['strain_rate']
    keyword.control_termination(endtim=endtim)
    sampling_number = (options['n_steps_coeff'] * options['size_coeff'] * options['strain_coeff'])
    if options['sim_type'] == 'implicit':
        keyword.control_implicit_general(imflag=1,
                                         dt0=endtim/sampling_number)
        keyword.control_implicit_auto(dtmin=endtim/(sampling_number*100), dtmax=endtim*2/sampling_number, iteopt=25)
        keyword.control_contact(shlthk=2)
        iacc=1
        if options['elem_type']==2:
            iacc=0
        keyword.control_accuracy(osu=1, inn=2, iacc=iacc)
        keyword.control_shell(istupd=0, psstupd=0, irnxx=-2, miter=2, nfail1=1, nfail4=1, esort=2)
        keyword.control_implicit_solution(dctol=5e-5, ectol=5e-4)
        keyword.control_implicit_solver()
        keyword.control_implicit_dynamics()

    else:
        keyword.control_timestep(dt2ms=0.0, tssfac=0.9)
        keyword.control_shell(istupd=0)

    keyword.comment_block('Database')
    keyword.database_glstat(dt=endtim / sampling_number)
    keyword.database_binary_d3_plot(dt=endtim / (50 * options['size_coeff'] * options['strain_coeff']))
    keyword.database_nodout(dt=keyword.endtim / sampling_number)
    keyword.database_nodfor(dt=keyword.endtim / sampling_number)
    keyword.database_spcforc(dt=keyword.endtim / sampling_number)
    keyword.database_bndout(dt=keyword.endtim / sampling_number)

    keyword.comment_block('Material, sections and parts')
    if material['mat_type'] == 'mat24':
        material['e']
        if material['stress'] != None:
            keyword.define_curve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=material['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        elif material['rate_mod'] != None:
            keyword.mat24_rate(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                               etan=material['etan'], ro=material['ro'], pr=material['pr'],
                               str_mod=material['rate_mod'], soften = material['soften'])  # DefaultParams, check LSKey
        elif material['soften']==True:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            keyword.define_curve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=mat_dict['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        else:
            keyword.mat24(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                          etan=material['etan'], pr=material['pr'], c=material['rate_c'], p=material['rate_c'],
                          ro=material['ro'])

    elif material['mat_type'] == 'mat181':
        if material['strain'] == None:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            mat_dict={'strain':list(-1*np.array(mat_dict['strain'])[::-1]) + list(mat_dict['strain'][1:]),
                      'stress':list(-1*np.array(mat_dict['stress'])[::-1]) + list(mat_dict['stress'][1:])}
            keyword.defineCurve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
        else:
            keyword.defineCurve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
        keyword.mat181(mid=1, youngs=mat_dict['E'], ro=material['ro'], lcid=100)

    keyword.mat_null(mid=2, e = material['e'])
    #keyword.mat24(mid=2, e=material['e']/2, sigy=0.0001, fail=material['matfail'], etan=0.00001)


    mesh_geometry.set_csa_sigma(options['csa_sigma'])
    mesh_geometry.set_tt_sigma(options['tt_sigma'])
    mesh_geometry.set_rho(rho=rho, phi=phi)

    if phi != 1.0: # If not only beams
        keyword.element_shell(mesh_geometry.shell_elements)
        if options['elem_type'] == 2 or options['elem_type'] == 10:
            keyword.control_hourglass(ihg=4, qh=0.05)
        elif abs(options['elem_type']) == 16:
            keyword.control_hourglass(ihg=8, qh=0.1)

        for surfs in mesh_geometry.surfs.values():
            keyword.section_shell(secid=surfs.id_, t1=surfs.tt, elform=options['elem_type'], nip=options['shell_nip'])
            if surfs.slave==True:
                keyword.part_contact(pid=surfs.id_, secid=surfs.id_, mid=2,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])
            else:
                keyword.part_contact(pid=surfs.id_, secid=surfs.id_, mid=1,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

    if phi !=0.0:
        mesh_geometry.set_beam_shape(options['beam_shape'])
        if options['beam_cs_shape'] == 'tri':
            keyword.mat28(mid=4, e=material['e'], sigy=material['sigy'], etan=material['etan'], pr=material['pr'],
                          ro=material['ro'])
            keyword.element_beam_section07_orientation(mesh_geometry.beam_elements)
            beam_elform = 2
            for beam in mesh_geometry.beams.values():
                mid=4
                if beam.slave == True:
                    mid=2
                keyword.section_beam(secid=beam.id_, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

        elif options['beam_cs_shape'] == 'round':
            keyword.element_beam_thickness_orientation(mesh_geometry.beam_elements)
            beam_elform = 1
            for beam in mesh_geometry.beams.values():
                mid=1
                if beam.slave == True:
                    mid = 2
                keyword.section_beam(secid=beam.id_, csa = beam.csa, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])


    if options['sim_type'] == 'implicit':
        #keyword.contact_automatic_single_surface_mortar_id(cid=5200001+i, ssid=5000001+i, sstyp=2, ignore=1)
        keyword.contact_automatic_single_surface_mortar_id(cid=5200001, ssid=0, sstyp=2, ignore=1)
    elif options['sim_type'] == 'explicit':
        keyword.set_part_list(sid=99,
                              pid_list=[solid.id_ for solid in mesh_geometry.solids.values()])
        keyword.contact_automatic_single_surface_id(cid=5200001, ssid=99, sstyp=6,
                                                 ignore=1, igap=2, snlog=1)

    if options['airbag'] == True:
        for i, volume in enumerate(
                mesh_geometry.tessellation.polyhedrons.values()):  # volume = list(mesh_geometry.tessObject.polyhedrons.values())[0]
            keyword.set_part_list(sid=5000001 + i,
                                  pid_list=[abs(surface) * 10 + mesh_geometry.surf_num_offset for surface in
                                            volume.faces])
        keyword.database_abstat(dt=keyword.endtim / sampling_number)
        number_of_polyhedons = i
        for i in range(number_of_polyhedons):
            keyword.airbag_adiabatic_gas_model(abid=5100001 + i, sid=5000001 + i)
    ##########################################################################
    if options['pert_nodes'] != 0.0:
        keyword.pertubation_node(options['pert_nodes'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    if options['pert_shell'] != 0.0:
        keyword.pertubationShell(options['pert_shell'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    ##########################################################################



    #def_gradient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.2]])
    keyword.node(mesh_geometry.nodes)
    BCs = BoundaryConditions(keyword, mesh_geometry)
    if options['sim_type'] == 'implicit':
        keyword.disp_type = 'disp'

    elif options['sim_type'] == 'explicit':
        keyword.disp_type = 'vel'
        keyword.mat24(mid=3, e=1e7, ro=1e-7, pr=0.0, sigy=1e9)
        keyword.element_solid(mesh_geometry.solid_elements)
        keyword.section_solid(secid=3, elform=2)
        for solid in mesh_geometry.solids.values():
            keyword.part(pid=solid.id_, secid=3, mid=3)

    if options['return_copy']==True:
       return BCs


    if options['sim_type'] == 'implicit':
        keyword = BCs.periodic_linear_local(def_gradient)
        keyword.end_key()
        keyword.write_key()
    elif options['sim_type'] == 'explicit':
        keyword = BCs.periodic_multiple_global(def_gradient)
        keyword.end_key()
        keyword.write_key()
##########################################################################

##########################################################################
    ##########################################################################
    if options['run'] == True: #Move to keyword_file?
        rem_working_folder = os.getcwd()
        os.chdir(keyword.model_file_name.rsplit('\\',1)[0])
        SOLVER = r'C:\Program Files\LSTC\LS-DYNA\ls-dyna_smp_d_R10.0_winx64_ifort160.exe'
        INPUT = model_file_name
        MEMORY = str(50)
        NCPU = str(1)
        subprocess.Popen('"{}" i={} ncpu={} memory={}m'.format(SOLVER, INPUT, NCPU, MEMORY))
        os.chdir(rem_working_folder)
    #return readASCI.readrwforc(workingFolder=workingFolder)

def non_periodic_template(tessellation, model_file_name, def_gradient, rho=0.05, phi=0.0, material_data={}, **kwargs):
    options = {
        'elem_type': 16,
        'strain_rate':  1.0,
        'size_coeff': 1.0,
        'strain_coeff': 1.0,
        'n_steps_coeff':500,
        'pert_nodes': 0.0,
        'pert_shell': 0.0,
        'tt_sigma':0.0,
        'csa_sigma': 0.0,
        'run': False,
        'return_copy': False,
        'sim_type':'implicit',
        'airbag':False,
        'beam_shape':'straight', #'marvi
        'beam_cs_shape':'round', #'tri'
        'shell_nip':7,
        'side_plates': ['x', 'y', 'z'],
        'overhang':0.0,
        'dt2ms':0.0
    }
    options.update(kwargs)
    material = {
        'e':1500,
        'sigy':25.0,
        'etan':1.0,
        'pr':0.3,
        'ro':9.2e-10,
        'matfail':2.0,
        'soften':False,
        'stress':None,
        'strain':None,
        'rate_c':0.0,
        'rate_p':0.0,
        'rate_mod':None, #[slope1, slope2, base_rate, break_rate], eg. [0.075, 0.2, 1e-3, 3e1]
        'mat_type':'mat24', #'mat181'
        'fs':0.76,
        'fd':0.76,
        'dc':0.0
    }
    material.update(material_data)

    mesh_geometry = mm.FoamModel(tessellation) # mesh_geometry = LSDynaPerGeom(perTessGeometry, debug=True) #mesh_geometry = LSDynaPerGeom(tessellation, debug=True)
    keyword = kw.Keyword(model_file_name)# model_file_name = r'H:\thesis\periodic\representative\S05R1\ID1\testKey.key'
    keyword.comment_block('Control')
    keyword.control_structured()
    endtim = options['strain_coeff']*options['size_coeff'] / options['strain_rate']
    keyword.control_termination(endtim=endtim)
    sampling_number = (options['n_steps_coeff'] * options['size_coeff'] * options['strain_coeff'])
    if options['sim_type'] == 'implicit':
        keyword.control_implicit_general(imflag=1,
                                         dt0=endtim/sampling_number)
        keyword.control_implicit_auto(dtmin=endtim/(sampling_number*100), dtmax=endtim*2/sampling_number, iteopt=25)
        keyword.control_contact(shlthk=2)
        iacc=1
        if options['elem_type']==2:
            iacc=0
        keyword.control_accuracy(osu=1, inn=2, iacc=iacc)
        keyword.control_shell(istupd=4, psstupd=0, irnxx=-2, miter=2, nfail1=1, nfail4=1, esort=2)
        keyword.control_implicit_solution(dctol=5e-5, ectol=5e-4)
        keyword.control_implicit_solver()
        keyword.control_implicit_dynamics()

    else:
        keyword.control_timestep(dt2ms=options['dt2ms'], tssfac=0.9)
        keyword.control_shell(istupd=4, psstupd=-99)

    keyword.comment_block('Database')
    keyword.database_glstat(dt=endtim / sampling_number)
    keyword.database_binary_d3_plot(dt=endtim / (30 * options['size_coeff'] * options['strain_coeff']))
    keyword.database_nodout(dt=keyword.endtim / sampling_number)
    keyword.database_nodfor(dt=keyword.endtim / sampling_number)
    keyword.database_spcforc(dt=keyword.endtim / sampling_number)
    keyword.database_bndout(dt=keyword.endtim / sampling_number)

    keyword.comment_block('Material, sections and parts')

    if material['mat_type'] == 'mat24':
        material['e']
        if material['stress'] != None:
            keyword.define_curve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=material['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        elif material['rate_mod'] != None:
            keyword.mat24_rate(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                               etan=material['etan'], ro=material['ro'], pr=material['pr'],
                               str_mod=material['rate_mod'], soften = material['soften'])  # DefaultParams, check LSKey
        elif material['soften']==True:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            keyword.define_curve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=mat_dict['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        else:
            keyword.mat24(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                          etan=material['etan'], pr=material['pr'], c=material['rate_c'], p=material['rate_c'],
                          ro=material['ro'])

    elif material['mat_type'] == 'mat181':
        if material['strain'] == None:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            mat_dict={'strain':list(-1*np.array(mat_dict['strain'])[::-1]) + list(mat_dict['strain'][1:]),
                      'stress':list(-1*np.array(mat_dict['stress'])[::-1]) + list(mat_dict['stress'][1:])}
            keyword.defineCurve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
        else:
            keyword.defineCurve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
        keyword.mat181(mid=1, youngs=mat_dict['e'], ro=material['ro'], lcid=100)

    #keyword.mat_null(mid=2, e = material['e'])
    keyword.mat24(mid=2, e=material['e']/30, sigy=0.0001, fail=0.0, etan=0.00001)


    mesh_geometry.set_csa_sigma(options['csa_sigma'])
    mesh_geometry.set_tt_sigma(options['tt_sigma'])
    mesh_geometry.set_rho(rho=rho, phi=phi)


    if phi != 1.0: # If not only beams
        keyword.element_shell(mesh_geometry.shell_elements)
        if options['elem_type'] == 2 or options['elem_type'] == 10:
            keyword.control_hourglass(ihg=4, qh=0.05)
        elif abs(options['elem_type']) == 16:
            keyword.control_hourglass(ihg=8, qh=0.1)

        for surf in mesh_geometry.surfs.values():
            keyword.section_shell(secid=surf.id_, t1=surf.tt, elform=options['elem_type'], nip=options['shell_nip'])
            if surf.slave==True:
                keyword.part_contact(pid=surf.id_, secid=surf.id_, mid=2,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])
            else:
                keyword.part_contact(pid=surf.id_, secid=surf.id_, mid=1,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

    if phi !=0.0: #If not only surfaces
        mesh_geometry.set_beam_shape(options['beam_shape'])
        if options['beam_cs_shape'] == 'tri':
            keyword.mat28(mid=4, e=material['e'], sigy=material['sigy'], etan=material['etan'], pr=material['pr'],
                          ro=material['ro'])
            keyword.element_beam_section07_orientation(mesh_geometry.beam_elements)
            beam_elform = 2
            for beam in mesh_geometry.beams.values():
                mid=4
                if beam.slave == True:
                    mid=2
                keyword.section_beam(secid=beam.id_, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

        elif options['beam_cs_shape'] == 'round':
            keyword.element_beam_thickness_orientation(mesh_geometry.beam_elements)
            beam_elform = 1
            for beam in mesh_geometry.beams.values():
                mid=1
                if beam.slave == True:
                    mid = 2
                keyword.section_beam(secid=beam.id_, csa = beam.csa, elform=beam_elform)
                keyword.part_contact(pid=beam.id_, secid=beam.id_, mid=mid,
                                     fs=material['fs'], fd=material['fd'], dc=material['dc'])

    if len(options['side_plates']) == 1 and options['overhang'] == 0.0:
        plane_map={'x':0, 'y':1, 'z':2}
        options['overhang'] = mesh_geometry.tessellation.domain_size[plane_map[options['side_plates'][0]]]/2

    mesh_geometry.create_side_elements(options['side_plates'], overhang=options['overhang'])
    side_parts = mesh_geometry.find_side_surfs(options['side_plates'])
    keyword.set_part_list(sid=99, pid_list=[side_part for side_part in side_parts if side_part != []])
    for side_part in side_parts:
        if side_part != []:
            surf = mesh_geometry.surfs[side_part[0]]
            keyword.section_shell(secid=surf.id_, t1=0.1, elform=options['elem_type'], nip=options['shell_nip'])
            keyword.part_contact(pid=surf.id_, secid=surf.id_, mid=2,
                                 fs=material['fs'], fd=material['fd'], dc=material['dc'])
            keyword.element_shell_offset([mesh_geometry.shell_elements[surf.elem_ids[0]]], offset=-0.05)


    if options['sim_type'] == 'implicit':
        #keyword.contact_automatic_single_surface_mortar_id(cid=5200001+i, ssid=5000001+i, sstyp=2, ignore=1)
        keyword.contact_automatic_single_surface_mortar_id(cid=5200001, ssid=99, sstyp=6, ignore=1)
    elif options['sim_type'] == 'explicit':

        keyword.contact_automatic_single_surface_id(cid=5200001, ssid=99, sstyp=6,
                                                 ignore=1, igap=2, snlog=1)

    if options['airbag'] == True:
        for i, volume in enumerate(
                mesh_geometry.tessellation.polyhedrons.values()):  # volume = list(mesh_geometry.tessObject.polyhedrons.values())[0]
            keyword.set_part_list(sid=5000001 + i,
                                  pid_list=[abs(surface) * 10 + mesh_geometry.surf_num_offset for surface in
                                            volume.faces])
        keyword.database_abstat(dt=keyword.endtim / sampling_number)
        number_of_polyhedons = i
        for i in range(number_of_polyhedons):
            keyword.airbag_adiabatic_gas_model(abid=5100001 + i, sid=5000001 + i)
    ##########################################################################
    if options['pert_nodes'] != 0.0:
        keyword.pertubation_node(options['pert_nodes'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    if options['pert_shell'] != 0.0:
        keyword.pertubationShell(options['pert_shell'], nsid=0, cmp=1, xwl=mesh_geometry.tessellation.domain_size[0]/10,
                                 ywl=0, zwl=0)
    ##########################################################################



    #def_gradient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.2]])
    mesh_geometry.delete_massless_nodes()
    keyword.node(mesh_geometry.nodes)
    BCs = BoundaryConditions(keyword, mesh_geometry)
    if options['sim_type'] == 'implicit':
        keyword.disp_type = 'disp'

    elif options['sim_type'] == 'explicit':
        keyword.disp_type = 'vel'

    if options['return_copy']==True:
       return BCs


    if options['sim_type'] == 'implicit':
        keyword = BCs.non_periodic(def_gradient, phi=phi, soft = 1)
        keyword.end_key()
        keyword.write_key()
    elif options['sim_type'] == 'explicit':
        keyword = BCs.non_periodic(def_gradient, phi=phi, soft = 1)
        keyword.end_key()
        keyword.write_key()
##########################################################################

##########################################################################
    ##########################################################################
    if options['run'] == True: #Move to keyword_file?
        rem_working_folder = os.getcwd()
        os.chdir(keyword.model_file_name.rsplit('\\',1)[0])
        SOLVER = r'C:\Program Files\LSTC\LS-DYNA\ls-dyna_smp_d_R10.0_winx64_ifort160.exe'
        INPUT = model_file_name
        MEMORY = str(50)
        NCPU = str(1)
        subprocess.Popen('"{}" i={} ncpu={} memory={}m'.format(SOLVER, INPUT, NCPU, MEMORY))
        os.chdir(rem_working_folder)
    #return readASCI.readrwforc(workingFolder=workingFolder)

def solid_elements_template(def_gradient, model_file_name, material_data={}, domain_size = np.array([1.0, 1.0, 1.0]), elem_size = 1.0, **kwargs):
    options = {
        'elem_type': 2,
        'strain_rate': 1.0,
        'size_coeff': 1.0,
        'strain_coeff': 1.0,
        'n_steps_coeff': 500,
        'run': False,
        'return_copy': False,
        'sim_type': 'implicit',
        'side_plates': ['x', 'y', 'z'],
        'overhang': 0.0,
        'dt2ms': 0.0
    }
    options.update(kwargs)
    material = {
        'e': 100,
        'sigy': 1.0,
        'etan': 1.0,
        'pr': 0.3,
        'ro': 3.2e-11,
        'matfail': 2.0,
        'soften': False,
        'stress': None,
        'strain': None,
        'rate_c': 0.0,
        'rate_p': 0.0,
        'rate_mod': None,  # [slope1, slope2, base_rate, break_rate], eg. [0.075, 0.2, 1e-3, 3e1]
        'mat_type': 'mat24',  # 'mat181'
        'fs': 0.0,
        'fd': 0.0,
        'dc': 0.0
    }
    material.update(material_data)

    mesh_geometry = mm.SolidModel(domain_size=domain_size, elem_size = elem_size)
    if len(options['side_plates']) == 1 and options['overhang'] == 0.0:
        plane_map={'x':0, 'y':1, 'z':2}
        options['overhang'] = mesh_geometry.domain_size[plane_map[options['side_plates'][0]]]/2
    mesh_geometry.create_side_elements(options['side_plates'], options['overhang'])
    keyword = kw.Keyword(model_file_name)

    keyword.comment_block('Control')
    keyword.control_structured()
    endtim = options['strain_coeff'] * options['size_coeff'] / options['strain_rate']
    keyword.control_termination(endtim=endtim)
    sampling_number = (options['n_steps_coeff'] * options['size_coeff'] * options['strain_coeff'])
    if options['sim_type'] == 'implicit':
        keyword.control_implicit_general(imflag=1,
                                         dt0=endtim / sampling_number)
        keyword.control_implicit_auto(dtmin=endtim / (sampling_number * 100), dtmax=endtim * 2 / sampling_number,
                                      iteopt=25)
        keyword.control_contact(shlthk=2)
        iacc = 1
        if options['elem_type'] == 2:
            iacc = 0
        keyword.control_accuracy(osu=1, inn=2, iacc=iacc)
        keyword.control_shell(istupd=4, psstupd=0, irnxx=-2, miter=2, nfail1=1, nfail4=1, esort=2)
        keyword.control_implicit_solution(dctol=5e-5, ectol=5e-4)
        keyword.control_implicit_solver()
        keyword.control_implicit_dynamics()

    else:
        keyword.control_timestep(dt2ms=options['dt2ms'], tssfac=0.9)
        keyword.control_shell(istupd=4, psstupd=-99)

    keyword.comment_block('Database')
    keyword.database_glstat(dt=endtim / sampling_number)
    keyword.database_binary_d3_plot(dt=endtim / (30 * options['size_coeff'] * options['strain_coeff']))
    keyword.database_nodout(dt=keyword.endtim / sampling_number)
    keyword.database_nodfor(dt=keyword.endtim / sampling_number)
    keyword.database_spcforc(dt=keyword.endtim / sampling_number)
    keyword.database_bndout(dt=keyword.endtim / sampling_number)

    if material['mat_type'] == 'mat24':
        material['e']
        if material['stress'] != None:
            keyword.define_curve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=material['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        elif material['rate_mod'] != None:
            keyword.mat24_rate(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                               etan=material['etan'], ro=material['ro'], pr=material['pr'],
                               str_mod=material['rate_mod'], soften = material['soften'])  # DefaultParams, check LSKey
        elif material['soften']==True:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            keyword.define_curve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=mat_dict['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        else:
            keyword.mat24(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                          etan=material['etan'], pr=material['pr'], c=material['rate_c'], p=material['rate_c'],
                          ro=material['ro'])

    elif material['mat_type'] == 'mat181':
        if material['strain'] == None:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            mat_dict={'strain':list(-1*np.array(mat_dict['strain'])[::-1]) + list(mat_dict['strain'][1:]),
                      'stress':list(-1*np.array(mat_dict['stress'])[::-1]) + list(mat_dict['stress'][1:])}
            keyword.defineCurve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
        else:
            keyword.defineCurve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
        keyword.mat181(mid=1, youngs=mat_dict['E']/30, ro=material['ro'], lcid=100)

    #keyword.mat_null(mid=2, e = material['e'])
    keyword.mat24(mid=2, e=material['e']/10, sigy=0.0001, fail=0.0, etan=0.00001)


    keyword.element_solid(mesh_geometry.solid_elements)
    for solid in mesh_geometry.solids.values():
        keyword.section_solid(secid=solid.id_, elform=options['elem_type'])
        keyword.part_contact(pid=solid.id_, secid=solid.id_, mid=1,
                             fs=material['fs'], fd=material['fd'], dc=material['dc'])

    keyword.element_shell_offset(mesh_geometry.shell_elements, offset=-0.05)

    for surf in mesh_geometry.surfs.values():
        keyword.section_shell(secid=surf.id_, t1=0.1, elform=options['elem_type'])
        keyword.part_contact(pid=surf.id_, secid=surf.id_, mid=2,
                             fs=material['fs'], fd=material['fd'], dc=material['dc'])

    side_parts = mesh_geometry.find_side_surfs(options['side_plates'])
    keyword.set_part_list(sid=99, pid_list=[side_part[0] for side_part in side_parts if side_part != []])

    keyword.node(mesh_geometry.nodes)
    keyword.disp_type = 'disp'
    #Boundary conditions
    BCs = BoundaryConditions(keyword, mesh_geometry)
    keyword = BCs.non_periodic(def_gradient, surf_contact=False, soft=1)
    keyword.end_key()
    keyword.write_key()

def single_elements_template(def_gradient, model_file_name, material_data={}, domain_size = np.array([1.0, 1.0, 1.0]), elem_size = 1.0, **kwargs):
    options = {
        'elem_type': 2,
        'strain_rate': 1.0,
        'size_coeff': 1.0,
        'strain_coeff': 1.0,
        'n_steps_coeff': 500,
        'run': False,
        'return_copy': False,
        'sim_type': 'implicit',
        'side_plates': [],
        'overhang': 0.0,
        'dt2ms': 0.0
    }
    options.update(kwargs)
    material = {
        'e': 1500,
        'sigy': 25.0,
        'etan': 1.0,
        'pr': 0.3,
        'ro': 9.2e-10,
        'matfail': 2.0,
        'soften': False,
        'stress': None,
        'strain': None,
        'rate_c': 0.0,
        'rate_p': 0.0,
        'rate_mod': None,  # [slope1, slope2, base_rate, break_rate], eg. [0.075, 0.2, 1e-3, 3e1]
        'mat_type': 'mat24',  # 'mat181'
        'fs': 0.76,
        'fd': 0.76,
        'dc': 0.0
    }
    material.update(material_data)

    mesh_geometry = mm.SolidModel(domain_size=domain_size, elem_size = elem_size)
    keyword = kw.Keyword(model_file_name)

    keyword.comment_block('Control')
    keyword.control_structured()
    endtim = options['strain_coeff'] * options['size_coeff'] / options['strain_rate']
    keyword.control_termination(endtim=endtim)
    sampling_number = (options['n_steps_coeff'] * options['size_coeff'] * options['strain_coeff'])
    if options['sim_type'] == 'implicit':
        keyword.control_implicit_general(imflag=1,
                                         dt0=endtim / sampling_number)
        keyword.control_implicit_auto(dtmin=endtim / (sampling_number * 100), dtmax=endtim * 2 / sampling_number,
                                      iteopt=25)
        keyword.control_contact(shlthk=2)
        iacc = 1
        if options['elem_type'] == 2:
            iacc = 0
        keyword.control_accuracy(osu=1, inn=2, iacc=iacc)
        keyword.control_shell(istupd=4, psstupd=0, irnxx=-2, miter=2, nfail1=1, nfail4=1, esort=2)
        keyword.control_implicit_solution(dctol=5e-5, ectol=5e-4)
        keyword.control_implicit_solver()
        keyword.control_implicit_dynamics()

    else:
        keyword.control_timestep(dt2ms=options['dt2ms'], tssfac=0.9)

    keyword.comment_block('Database')
    keyword.database_glstat(dt=endtim / sampling_number)
    keyword.database_binary_d3_plot(dt=endtim / (30 * options['size_coeff'] * options['strain_coeff']))
    keyword.database_nodout(dt=keyword.endtim / sampling_number)
    keyword.database_nodfor(dt=keyword.endtim / sampling_number)
    keyword.database_spcforc(dt=keyword.endtim / sampling_number)
    keyword.database_bndout(dt=keyword.endtim / sampling_number)

    if material['mat_type'] == 'mat24':
        material['e']
        if material['stress'] != None:
            keyword.define_curve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=material['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        elif material['rate_mod'] != None:
            keyword.mat24_rate(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                               etan=material['etan'], ro=material['ro'], pr=material['pr'],
                               str_mod=material['rate_mod'], soften = material['soften'])  # DefaultParams, check LSKey
        elif material['soften']==True:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            keyword.define_curve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
            keyword.mat24(mid=1, e=material['e'], sigy=mat_dict['stress'][0], fail=material['matfail'],
                          ro=material['ro'], lcss=100)
        else:
            keyword.mat24(mid=1, e=material['e'], sigy=material['sigy'], fail=material['matfail'],
                          etan=material['etan'], pr=material['pr'], c=material['rate_c'], p=material['rate_c'],
                          ro=material['ro'])

    elif material['mat_type'] == 'mat181':
        if material['strain'] == None:
            mat_dict=keyword.soften_yield_curve(youngs=material['e'], sigy=material['sigy'], etan=material['etan'])
            mat_dict={'strain':list(-1*np.array(mat_dict['strain'])[::-1]) + list(mat_dict['strain'][1:]),
                      'stress':list(-1*np.array(mat_dict['stress'])[::-1]) + list(mat_dict['stress'][1:])}
            keyword.defineCurve(lcid=100, abscissas=mat_dict['strain'], ordinates=mat_dict['stress'])
        else:
            keyword.defineCurve(lcid=100, abscissas=material['strain'], ordinates=material['stress'])
        keyword.mat181(mid=1, youngs=mat_dict['E'], ro=material['ro'], lcid=100)

    #keyword.mat_null(mid=2, e = material['e'])
    keyword.mat24(mid=2, e=material['e']/30, sigy=0.0001, fail=0.0, etan=0.00001)
    keyword.mat24(mid=3, e=material['e']*10, sigy=100000, fail=0.0, etan=1000)

    keyword.element_solid(mesh_geometry.solid_elements)
    for solid in mesh_geometry.solids.values():
        keyword.section_solid(secid=solid.id_, elform=options['elem_type'])
        keyword.part_contact(pid=solid.id_, secid=solid.id_, mid=1,
                             fs=material['fs'], fd=material['fd'], dc=material['dc'])

    keyword.node(mesh_geometry.nodes)
    keyword.disp_type = 'vel'
    #Boundary conditions
    BCs = BoundaryConditions(keyword, mesh_geometry)
    keyword = BCs.solid_element_single(def_gradient)
    keyword.end_key()
    keyword.write_key()



#def_gradient = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.2]])
os.chdir(r'H:\thesis\periodic\representative\S05R1\ID1')
model_file_name = 'testKey.key'
#periodic_template(tessellation, model_file_name, def_gradient, sim_type='implicit', strain_coeff = 0.2, strain_rate = 1e2, size_coeff = 0.5)