import tessellations as ts
import copy
import importlib
from collections import namedtuple
import numpy as np
import math
#importlib.reload(ts)
#folderName = r'H:\thesis\periodic\representative\S05R1\ID1'
#mesh_file_name = folderName + r'\\test'
#tessellation  = ts.Tessellation(folderName + r'\\nfrom_morpho-id1.tess')
#tessellation .regularize(n=int(len(self.edges.keys())/2))
#tessellation .mesh_file_name=mesh_file_name
#tessellation .mesh2D(elem_size=0.02)

class NodeClass(object):
    def __init__(self, id_, coord):
        self.id_ = id_
        self.coord = coord
        self.master_to = []
        self.slave_to = []

class ShellElementClass(object):
    def __init__(self, nodes, id_, parent, node_ids):
        self.nodes = nodes
        self.id_ = id_
        self.parent = parent
        self.node_ids = node_ids
        self.area = self.find_area()

    def find_area(self):
        coords=np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        v1=coords[2]-coords[0]
        v2=coords[3]-coords[1]
        A1=np.linalg.norm(np.cross(v1, v2))/2.
        return abs(A1)

class BeamElementClass(object):
    def __init__(self, nodes, id_, parent, node_ids):
        self.nodes = nodes
        self.id_ = id_
        self.parent = parent
        self.node_ids = node_ids
        self.csa = None
        self.orientation = []
        self.length = self.find_length()

    def find_length(self):
        coords = np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        v1 = coords[1] - coords[0]
        L1 = np.linalg.norm(v1)
        return L1

    def midpoint(self):
        return np.mean(np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]]), axis=0)

    def find_volume(self, csa = None):
        if self.csa == None and csa == None:
            raise Exception('Cross sectional area not set')
        elif csa != None:
            self.csa = csa
            return self.find_volume()
        else:
            return self.length * self.csa

class SolidElementClass(object):
    def __init__(self, nodes, id_, parent, node_ids):
        self.nodes = nodes
        self.id_ = id_
        self.parent = parent
        self.node_ids = node_ids

class SurfPartClass(object):
    def __init__(self, shell_elements, id_, elem_ids):
        self.shell_elements = shell_elements
        self.id_ = id_
        self.elem_ids = elem_ids
        self.tt = None
        self.area = self.find_area()
        self.slave = False
    def find_area(self):
        return sum([self.shell_elements[elem_id].area for elem_id in self.elem_ids])

    def find_volume(self, tt=None):
        if self.tt == None and tt == None:
            raise Exception('Element thickness not set')
        elif tt != None:
            self.tt = tt
            return self.find_volume()
        else:
            return self.area * self.tt

    def set_volume(self, volume):
        self.tt = volume/self.area

class BeamPartClass(object):
    def __init__(self, beam_elements, id_, elem_ids):
        self.beam_elements = beam_elements
        self.id_ = id_
        self.elem_ids = elem_ids
        self.csa = None
        self.slave = False
        self.length = self.find_length()

    def find_length(self):
        return sum([self.beam_elements[elem_id].length for elem_id in self.elem_ids])

    def set_csa(self, csa):
        self.csa = csa
        for elem_id in self.elem_ids:
            self.beam_elements[elem_id].csa = csa

    def set_volume(self, volume, beam_shape = 'straight'):
        def marvi_scaling(x_):
            #between -0.5, 0.5
            return (5.45*(x_)**4 + 2.63*(x_)**2 + 1)
        if beam_shape == 'straight':
            self.set_csa(volume/self.find_length())
        elif beam_shape == 'marvi':
            # A = A_0*(5.45(x/L)**4 + 2.63(x/L)**2 + 1)
            #Volume = 1.28729*self.find_length()*A0
            A0 = volume/(1.28729*self.find_length())
            end_coord = list(self.beam_elements.values())[0].nodes[self.find_corner_node()].coord
            for elem_id in self.elem_ids: #elem_id = self.elem_ids[0]
                x_ = np.linalg.norm(self.beam_elements[elem_id].midpoint() - end_coord)/self.length - 0.5
                self.beam_elements[elem_id].csa = A0*marvi_scaling(x_)

    def set_beam_shape(self, csa=None, beam_shape = 'straight'):
        if self.csa == None and csa == None:
            raise Exception('Beam csa not set')
        elif csa != None:
            self.csa = csa

        def marvi_scaling(x_):
            #between -0.5, 0.5
            return (5.45*(x_)**4 + 2.63*(x_)**2 + 1)

        if beam_shape == 'straight':
            self.set_csa(self.csa)
        elif beam_shape == 'marvi':
            # A = A_0*(5.45(x/L)**4 + 2.63(x/L)**2 + 1)
            #Volume = 1.28729*self.find_length()*A0
            A0 = self.csa/1.28729
            end_coord = list(self.beam_elements.values())[0].nodes[self.find_corner_node()].coord
            for elem_id in self.elem_ids: #elem_id = self.elem_ids[0]
                x_ = np.linalg.norm(self.beam_elements[elem_id].midpoint() - end_coord)/self.length - 0.5
                self.beam_elements[elem_id].csa = A0*marvi_scaling(x_)

    def find_corner_node(self):
        distance_list = []
        for elem_id in self.elem_ids:
            for node_id in self.beam_elements[elem_id].node_ids:
                dist_from_origo = np.linalg.norm(self.beam_elements[elem_id].nodes[node_id].coord)
                distance_list.append([node_id, dist_from_origo])
        distance_list = np.array(distance_list)
        furthest_node = int(distance_list[np.argmax(distance_list[:,1]), 0])
        return furthest_node

    def find_volume(self):
        return sum([self.beam_elements[elem_id].find_volume() for elem_id in self.elem_ids])

class PeriodicLSDynaGeometry(object):
    '''Takes inn a tessGeom object that must have been meshed and have an assigned meshFileName'''
    def __init__(self, tessellation, debug=False):
        self.tessellation=copy.deepcopy(tessellation)
        #if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        self.shell_elements = self.load_mesh()
        self.surf_num_offset = 2000000
        self.beam_num_offset = 4000000
        self.solid_num_offset = 6000000
        self.last_element_key = max(self.shell_elements.keys())
        self.last_node_key = max(self.nodes.keys())
        self.rho = None
        self.phi = None
        self.surfs = self.find_surfs()
        self.vertex_to_node, self.node_to_vertex = self.find_vertex_node_map()
        self.nodes_on_edges = self.find_nodes_on_edges()
        if self.tessellation.periodic == True:
            self.find_vertex_nodes_periodicity()
            self.find_edge_nodes_periodicity()
            self.transfer_surfaces()
            self.beam_elements = self.find_beam_elements()
            self.beams = self.find_beams()


        if debug == False:
            #self.domain_size = self.tessellation.domain_size
            self.find_slave_surfs()
            self.find_slave_beams()
            self.beam_oriention = self.find_beam_oriention()
            self.solids = self.find_reference_elements()

    def load_mesh(self):
        #self.node_tup = namedtuple('node', ['n', 'coords'])
        #self.element_tup = namedtuple('element', ['elem', 'part', 'nodes'])
        #self.part_tup = namedtuple('part', ['pid', 'elems'])

        with open(self.tessellation.mesh_file_name.rsplit('.')[0] + '.key', 'r') as read_file:
            lines = read_file.readlines()
        keyword_ind = [i for i, line in enumerate(lines) if '*' in line]

        ################################################################################
        # Read Nodes
        ################################################################################
        node_keyword_ind = np.where(np.array(lines, dtype=object) == '*NODE\n')[0][0]
        node_last_index = keyword_ind[keyword_ind.index(node_keyword_ind) + 1]
        nodes = {}
        for line in lines[node_keyword_ind + 1:node_last_index]:
            if '$' not in line:
                if ',' in line:
                    id_ = int(line.replace('\n', '').split(',')[0])
                    coord = np.array([float(item) for item in line.replace('\n', '').split(',')[1:]])
                    nodes[id_] = NodeClass(id_, coord)
                elif isinstance(int(line.split()[0]), int):
                    id_ = int(line.replace('\n', '').split()[0])
                    coord =  np.array([float(item) for item in line.replace('\n', '').split()[1:]])
                    nodes[id_] = NodeClass(id_,coord)
                else:
                    raise Exception('Unexpected value in list of nodes')
        self.nodes = nodes
        ################################################################################
        # Read Elements
        ################################################################################
        elem_keyword_ind = np.where(np.array(lines, dtype=object) == '*ELEMENT_SHELL\n')[0]
        elem_last_ind = keyword_ind[keyword_ind.index(elem_keyword_ind[-1]) + 1]
        elements = {}
        for line in lines[elem_keyword_ind[0] + 1:elem_last_ind]: #line = lines[elem_keyword_ind[0] + 1:elem_last_ind][0]
            if '$' not in line and '*' not in line:
                if ',' in line:
                    id_ = int(line.replace('\n', '').split(',')[0])
                    parent = int(line.replace('\n', '').split(',')[1])
                    node_ids = [int(item) for item in line.replace('\n', '').split(',')[2:]]
                    elements[id_] = ShellElementClass(self.nodes, id_, parent, node_ids)
                elif isinstance(int(line.split()[0]), int):
                    id_ = int(line.replace('\n', '').split()[0])
                    parent = int(line.replace('\n', '').split()[1])
                    node_ids = [int(item) for item in line.replace('\n', '').split()[2:]]
                    elements[id_] = ShellElementClass(self.nodes, id_, parent, node_ids)
                else:
                    raise Exception('Unexpected value in list of shell_elements')
        return elements

    def find_surfs(self):
        part_list = set([element.parent for element in self.shell_elements.values()])
        part_dict = {}
        for part in part_list:
            part_dict[part] = SurfPartClass(self.shell_elements, part, [element.id_ for element in self.shell_elements.values() if element.parent == part])
        return part_dict

    def compare_arrays(self, arr0, arr1, rel_tol=1e-07, abs_tol=0.0):
        return all([math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(arr0, arr1)])

    def find_vertex_node_map(self):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        vertex_to_node={}
        for i, vertex in enumerate(self.tessellation.vertices.values()):
            vertex_to_node[vertex.id_] = i+1 #Works because gmsh node numbers after the input file
            if self.compare_arrays(vertex.coord, self.nodes[i + 1].coord) == False:
                raise Exception('Vertex {} and node {} location not equal'.format(vertex.id_, i+1))
            node_to_vertex = dict(zip(vertex_to_node.values(), vertex_to_node.keys()))
        if len(vertex_to_node) != len(self.tessellation.vertices.keys()):
            raise Exception('All vertexes not mapped')
        return vertex_to_node, node_to_vertex

    def find_nodes_on_edges(self):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        edge_dict = {}
        for edge in self.tessellation.edges.values():  # Have to consider two edges for connected surfaces
            surf_nodes = []
            for surf in edge.parents: #surf=edge.parents[0]
                temp_list = []
                for elem in self.surfs[surf * 10 + self.surf_num_offset].elem_ids:
                    temp_list.extend(self.shell_elements[elem].node_ids)
                surf_nodes.append(temp_list)
            nodes_on_edge = list(set(surf_nodes[0]).intersection(set(surf_nodes[1])))
            for node in nodes_on_edge:
                if self.compare_arrays(self.nodes[node].coord, edge.x0(), rel_tol=1e-08, abs_tol=1e-8):
                    origo_node = node
            filtered_nodes_on_edge=[origo_node]
            edge_vector=edge.vector()/np.linalg.norm(edge.vector())
            nodes_on_edge.remove(origo_node)
            for node in nodes_on_edge:
                node_pair_vector = self.nodes[node].coord - self.nodes[origo_node].coord
                norm_node_pair_vector = node_pair_vector/np.linalg.norm(node_pair_vector)
                if self.compare_arrays(abs(edge_vector), abs(norm_node_pair_vector), rel_tol=1e-08, abs_tol=1e-8):
                    filtered_nodes_on_edge.append(node)
            edge_dict[edge.id_] = filtered_nodes_on_edge
        return edge_dict

    def find_vertex_nodes_periodicity(self):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        for vertex in self.tessellation.vertices.values(): #vertex = list(self.tessellation.vertices.values())[0]
            if vertex.master_to != []:
                master_node_id=self.vertex_to_node[vertex.id_]
                for i in range(0, len(vertex.master_to), 4):
                    slave_node_id = self.vertex_to_node[vertex.master_to[i]]
                    self.nodes[master_node_id].master_to.extend([slave_node_id] + vertex.master_to[i + 1: i + 4])
                    self.nodes[slave_node_id].slave_to.extend([master_node_id] + vertex.master_to[i + 1: i + 4])

    def find_edge_nodes_periodicity(self):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        for edge in self.tessellation.edges.values(): #edge = self.tessellation.edges[1]
            if edge.master_to != []:
                master_nodes = self.nodes_on_edges[edge.id_]
                for i in range(0, len(edge.master_to), 5): #i=0
                    periodicity = np.array(edge.master_to[i+1:i+4])*self.tessellation.domain_size
                    slave_nodes = self.nodes_on_edges[edge.master_to[i]]
                    counter = 0
                    for master_node in master_nodes: #master_node = master_nodes[0]
                        if master_node not in self.node_to_vertex.keys():
                            for slave_node in slave_nodes:  #slave_node =  slave_nodes[5]
                                if self.compare_arrays(self.nodes[master_node].coord,
                                                       self.nodes[slave_node].coord - periodicity,
                                                       abs_tol=1e-08, rel_tol=1e-8):
                                    self.nodes[master_node].master_to.extend([slave_node]+edge.master_to[i+1:i+4])
                                    if self.nodes[slave_node].slave_to != []:
                                        raise Exception('Slave node {} referenced twice'.format(slave_node))
                                    self.nodes[slave_node].slave_to = [master_node]+edge.master_to[i+1:i+4]
                                    counter +=1
                    if counter != len(master_nodes)-2:
                        raise Exception('Unequal number of nodes in master edge {} '
                                        'and slave edge {}'.format(edge.id_,edge.master_to[i]))

    def transfer_surfaces(self):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
        for face in self.tessellation.faces.values(): #face = list(self.tessellation.faces.values())[0]
            if face.master_to != []:
                unit_periodicity = face.master_to[1:4]
                periodicity = np.array(unit_periodicity)*self.tessellation.domain_size
                master_part = self.surfs[face.id_ * 10 + self.surf_num_offset]
                master_part_edge_nodes = list(set(
                    [node for edge in self.tessellation.faces[face.id_].edges for node in
                     self.nodes_on_edges[abs(edge)]]))
                slave_part_id = face.master_to[0] * 10 + self.surf_num_offset
                slave_part = self.surfs[slave_part_id]
                slave_part_edge_nodes = list(set([node for edge in self.tessellation.faces[face.master_to[0]].edges for node in self.nodes_on_edges[abs(edge)]]))
                all_nodes_on_master_face=list(set([node for elem in master_part.elem_ids
                                                   for node in self.shell_elements[elem].node_ids]))
                all_nodes_on_slave_face = list(set([node for elem in slave_part.elem_ids
                                                    for node in self.shell_elements[elem].node_ids]))
                #Create new node numbering for all slave nodes. Re-assign edge nodes.
                new_node_map_dict = {}
                for master_node_id in all_nodes_on_master_face:
                    self.last_node_key += 1
                    new_node_map_dict[master_node_id] = self.last_node_key
                for master_node_id in master_part_edge_nodes:
                    for slave_node_id in slave_part_edge_nodes:
                        if self.compare_arrays(self.nodes[master_node_id].coord,
                                    self.nodes[slave_node_id].coord - periodicity):
                            new_node_map_dict[master_node_id] = slave_node_id


                for master_node_id in new_node_map_dict.keys():
                    if master_node_id not in master_part_edge_nodes:
                        slave_node_id = new_node_map_dict[master_node_id]
                        slave_coord =  self.nodes[master_node_id].coord + periodicity
                        self.nodes[master_node_id].master_to.append([slave_node_id]
                                                                        + unit_periodicity)
                        self.nodes[slave_node_id] = NodeClass(slave_node_id, slave_coord)
                        self.nodes[slave_node_id].slave_to = [master_node_id] + unit_periodicity

                for m_element in master_part.elem_ids: # m_element = master_part.elem_ids[0]
                    self.last_element_key +=1
                    new_node_list = []
                    for i, mnode in enumerate(self.shell_elements[m_element].node_ids): # mnode = self.shell_elements[m_element].node_ids[0]
                        new_node_list.append(new_node_map_dict[mnode])
                    #if len(new_node_list) != 4:
                    #    raise Exception('NewNodeList is too short! Investigate')
                    self.shell_elements[self.last_element_key] = ShellElementClass(
                        self.nodes, self.last_element_key, slave_part_id, new_node_list
                    )

                for element in slave_part.elem_ids:
                    del self.shell_elements[element]
                for node in all_nodes_on_slave_face:
                    if node not in slave_part_edge_nodes:
                        del self.nodes[node]
                self.surfs[slave_part_id] = SurfPartClass(self.shell_elements, slave_part_id,
                    [element.id_ for element in self.shell_elements.values() if element.parent == slave_part_id])
                self.surfs[slave_part_id].slave = True

                if set(slave_part_edge_nodes).intersection(
                        set([node for elem in self.surfs[slave_part_id].elem_ids for
                             node in self.shell_elements[elem].node_ids])) != set(slave_part_edge_nodes):
                    raise Exception('SlaveNodes not preserved')

    def find_beam_elements(self):
        beam_elements = {}
        for edge in self.tessellation.edges.values(): #edge = list(self.tessellation.edges.values())[0]
            example_surf = edge.parents[0]
            nodes_on_edge = self.nodes_on_edges[edge.id_]
            edge_center = np.mean(
                [self.tessellation.vertices[ver_id].coord for ver_id in self.tessellation.edges[edge.id_].verts],
                axis=0)
            edge_orientation = self.tessellation.faces[example_surf].find_barycenter() - edge_center
            for elem_id in self.surfs[example_surf * 10 + self.surf_num_offset].elem_ids: # elem_id =  self.surfs[example_surf  * 10 + self.surf_num_offset].elem_ids[0]
                intersecting_nodes = set(nodes_on_edge).intersection(set(self.shell_elements[elem_id].node_ids))
                if len(intersecting_nodes) >= 2:
                    self.last_element_key += 1
                    id_ = self.last_element_key
                    beam_elements[id_] = BeamElementClass(self.nodes, id_, edge.id_ *10 + self.beam_num_offset, list(intersecting_nodes))
                    beam_elements[id_].orientation = edge_orientation
        return beam_elements

    def find_beams(self):
        part_list = set([element.parent for element in self.beam_elements.values()])
        part_dict = {}
        for part in part_list:
            part_dict[part] = BeamPartClass(self.beam_elements, part, [element.id_ for element in self.beam_elements.values() if
                                                  element.parent == part])
            if self.tessellation.edges[int(part - self.beam_num_offset)/10].slave_to != []:
                part_dict[part].slave = True
        return part_dict

    def surface_area(self):
        return sum([part.area for part in self.surfs.values() if part.slave == False])

    def beam_length(self):
        return sum([part.length for part in self.beams.values() if part.slave == False])

    def surface_volume(self):
        return sum([part.find_volume() for part in self.surfs.values() if part.slave == False])

    def beam_volume(self):
        return sum([part.find_volume() for part in self.beams.values() if part.slave == False])

    def find_rho(self):
        bulk_volume=self.surface_volume() + self.beam_volume()
        total_volume = np.prod(self.tessellation.domain_size)
        return bulk_volume/total_volume

    def set_phi(self, phi=None, rho=None):
        if self.rho ==  None and rho == None:
            raise Exception('Relative density not defined')
        elif rho != None:
            self.rho = rho
        if self.phi == None and phi == None:
            raise Exception('Phi not defined')
        elif phi != None:
            self.phi = phi
        surface_area = self.surface_area()
        beam_length = self.beam_length()
        solid_volume = self.rho * np.prod(self.tessellation.domain_size)
        #(csa * beam_length + tt * surface_area) = solid_volume
        #(csa * beam_length/surface_area + tt) = solid_volume/surface_area
        ##tt = solid_volume/surface_area - csa * beam_length/surface_area
        #phi = beam_volume/solid_volume
        #phi = csa * beam_length/solid_volume
        ##csa = phi*solid_volume/beam_length
        csa = self.phi*solid_volume/beam_length
        tt = solid_volume/surface_area - csa * beam_length/surface_area
        for beam in self.beams.values():
            beam.set_csa(csa)
        for surf in self.surfs.values():
            surf.tt = tt

    def set_rho(self, rho=None, phi=None):
        if self.rho ==  None and rho == None:
            raise Exception('Relative density not defined')
        elif rho != None:
            self.rho = rho
        if self.phi == None and phi == None:
            raise Exception('Phi not defined')
        elif phi != None:
            self.phi = phi
        self.set_phi()

    def set_beam_shape(self, beam_shape):
        """'straingt', 'marvi'"""
        for beam in self.beams.values():
            beam.set_beam_shape(beam_shape=beam_shape)

    def surf_fraction(self):
        surf_volume = sum([part.find_volume() for part in self.surfs.values() if part.slave == False])
        solid_volume = surf_volume + sum([part.find_volume() for part in self.beams.values() if part.slave == False])
        return surf_volume/solid_volume

    def find_phi(self):
        surf_volume = sum([part.find_volume() for part in self.surfs.values() if part.slave == False])
        beam_volume = sum([part.find_volume() for part in self.beams.values() if part.slave == False])
        solid_volume = surf_volume + beam_volume
        return beam_volume/solid_volume

    def find_slave_surfs(self):
        return [part.id_ for part in self.surfs.values() if part.slave == True]

    def find_slave_beams(self):
        return [part.id_ for part in self.beams.values() if part.slave == True]
    ##########################################################################
    # Find node pairs and directions
    ##########################################################################
    def find_reference_elements(self, ref_element_size=0.1):
        elem_counter = 1
        self.solid_num_offset += 10
        ref_elem_dict = {}
        elem_counter += 1
        ref_loc = np.array([0., 0., 0.])
        node_list = []
        elementOffset =  np.array([[0, 0, 0], [ref_element_size, 0, 0], [ref_element_size, ref_element_size, 0], [0, ref_element_size, 0], [0, 0, ref_element_size], [ref_element_size, 0, ref_element_size], [ref_element_size, ref_element_size, ref_element_size], [0, ref_element_size, ref_element_size]])
        for offset in elementOffset:
            self.last_node_key += 1
            self.nodes[self.last_node_key] = NodeClass(self.last_node_key, ref_loc + offset)
            node_list.append(self.last_node_key)
            ref_elem_dict[elem_counter] = SolidElementClass(self.nodes, elem_counter, self.solid_num_offset, node_list)
        for i, location in enumerate(self.tessellation.domain_size): #location = self.domain_size[0]
            elem_counter += 1
            ref_loc = np.array([0., 0., 0.])
            ref_loc[i] = location
            node_list = []
            for offset in elementOffset:
                self.last_node_key += 1
                self.nodes[self.last_node_key] = NodeClass(self.last_node_key, ref_loc + offset)
            ref_elem_dict[elem_counter] = SolidElementClass(self.nodes, elem_counter, self.solid_num_offset, node_list)
        return ref_elem_dict

self=PeriodicLSDynaGeometry(tessellation=tessellation, debug=True)