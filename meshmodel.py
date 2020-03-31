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

#importlib.reload(ts)
#folderName = r'H:\thesis\linear\representative\S05R1\ID1'
#mesh_file_name = folderName + r'\\test'
#tessellation  = ts.Tessellation(folderName + r'\\nfrom_morpho-id1.tess')
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

    def on_plane(self, plane = None, loc = None):
        plane_map={'x':0, 'y':1, 'z':2}
        coords = np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        for coord in coords:
            if coord[plane_map[plane]] != loc:
                return False
        return True

    def incedent_to_plane(self, plane = None, loc = None):
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        coords = np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        for coord in coords:
            if coord[plane_map[plane]] == loc:
                return True
        return False

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

    def on_plane(self, plane = None, loc = None):
        plane_map={'x':0, 'y':1, 'z':2}
        coords = np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        for coord in coords:
            if coord[plane_map[plane]] != loc:
                return False
        return True

    def incedent_to_plane(self, plane = None, loc = None):
        plane_map={'x':0, 'y':1, 'z':2}
        coords = np.array([node.coord for node in [self.nodes[node_id] for node_id in self.node_ids]])
        for coord in coords:
            if coord[plane_map[plane]] == loc:
                return True
        return False

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

    def on_plane(self, plane, loc):
        for elem_id in self.elem_ids[:1]: # Only one element needed for check.
            if self.shell_elements[elem_id].on_plane(plane, loc) != True:
                return False
        return True

    def incident_to_plane(self, plane, loc):
        for elem_id in self.elem_ids:
            if self.shell_elements[elem_id].incident_to_plane(plane, loc) == True:
                return True
        return False

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

    def on_plane(self, plane, loc):
        for elem_id in self.elem_ids[:1]: # Only one element needed for check.
            if self.beam_elements[elem_id].on_plane(plane, loc) != True:
                return False
        return True

    def incident_to_plane(self, plane, loc):
        for elem_id in self.elem_ids:
            if self.beam_elements[elem_id].incident_to_plane(plane, loc) == True:
                return True
        return False

class LSDynaGeometry(object):
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
        self.nodes_on_edges = self.find_nodes_on_edges()
        self.beam_elements = self.find_beam_elements()
        self.beams = self.find_beams()
        if self.tessellation.periodic == True:
            self.vertex_to_node, self.node_to_vertex = self.find_vertex_node_map()
            self.find_vertex_nodes_periodicity()
            self.find_edge_nodes_periodicity()
            self.transfer_surfaces()
            self.beam_oriention = self.find_beam_oriention()
            self.solids = self.find_reference_elements()

        if self.tessellation.periodic == False:
            self.delete_elements_on_sides(save_corner_beams=False, save_side_beams=False)
        if debug == False:
            pass


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
        #if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
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

    def find_reference_elements(self, ref_element_size=0.1):
        if self.tessellation.periodic == False: raise Exception('Invalid action for current tesselation')
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

    ##########################################################################
    # Linear model
    ##########################################################################
    def find_corner_nodes(self):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        corner_nodes = [0] * 8
        x_size, y_size, z_size = list(self.tessellation.domain_size)
        corner_locs = [[0.0,0.0,0.0],
                       [x_size, 0.0, 0.0],
                       [x_size, y_size, 0.0],
                       [0.0, y_size, 0.0],
                       [0.0, 0.0, z_size],
                       [x_size, 0.0, z_size],
                       [x_size, y_size, z_size],
                       [0.0, y_size, z_size]]
        for node in self.nodes.values(): #node= list(self.nodes.values())[0]
            for i, corner_loc in enumerate(corner_locs):
                if self.compare_arrays(node.coord,  corner_loc):
                    corner_nodes[i] = node.id_
        if 0 in corner_nodes:
            raise Exception('Could not find all eight corner nodes')
        return corner_nodes

    def delete_elements_on_sides(self, planes=['x', 'y', 'z'], save_side_beams=False, save_corner_beams=False):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        del_elems=[]
        plane_map={'x':0, 'y':1, 'z':2}
        for plane in planes: #plane = 'x'
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]: #plane_loc=0.0
                for elem in self.shell_elements.values(): #elem = list(self.shell_elements.values())[0]
                    if elem.on_plane(plane, plane_loc):
                         del_elems.append(elem.id_)

        shell_nodes = np.array([self.shell_elements[del_elem].node_ids for del_elem in del_elems]).flatten()
        edge_nodes = np.array([edge_node for edge_nodes in self.nodes_on_edges.values() for edge_node in edge_nodes])
        for del_elem in set(del_elems):
            del self.shell_elements[del_elem]

        filtered_del_node_list=set(shell_nodes)-set(edge_nodes)
        for node_id in filtered_del_node_list:
            del self.nodes[node_id]

        self.surfs = self.find_surfs()

        # Beams

        del_elems = []
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        for elem in self.beam_elements.values(): #elem = list(self.beam_elements.values())[0]
            in_plane_threshold = 1
            if save_side_beams == True:
                in_plane_threshold = 2
            if save_corner_beams == True:
                in_plane_threshold = 3
            true_counter = []
            for plane in planes:
                for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                        true_counter.append(elem.on_plane(plane, plane_loc))
            if sum(true_counter) >= in_plane_threshold:
                del_elems.append(elem.id_)

        for del_elem in set(del_elems):
            del self.beam_elements[del_elem]

        self.beams = self.find_beams()

    def create_side_elements(self, sides=['x', 'y', 'z']):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        new_coord_systems = []
        part_id = int(max(self.surfs.keys()))
        self.corner_nodes = self.find_corner_nodes()
        if 'x' in [side.lower() for side in sides]:
            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[0], self.corner_nodes[3], self.corner_nodes[7], self.corner_nodes[4]]
            new_coord_systems.append([node_ids[0], node_ids[1], node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )

            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[1], self.corner_nodes[5], self.corner_nodes[6], self.corner_nodes[2]]
            new_coord_systems.append([node_ids[0], node_ids[1], node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )


        elif 'y' in [side.lower() for side in sides]:
            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[0], self.corner_nodes[4], self.corner_nodes[5], self.corner_nodes[1]]
            new_coord_systems.append([node_ids[0], node_ids[1], node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )

            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[2], self.corner_nodes[6], self.corner_nodes[7], self.corner_nodes[3]]
            new_coord_systems.append([node_ids[0], node_ids[1], node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )
        elif 'z' in [side.lower() for side in sides]:
            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[0], self.corner_nodes[1], self.corner_nodes[2], self.corner_nodes[3]]
            new_coord_systems.append([node_ids[0], node_ids[1],node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )

            self.last_element_key += 1
            part_id += 10
            node_ids = [self.corner_nodes[4], self.corner_nodes[7], self.corner_nodes[6], self.corner_nodes[5]]
            new_coord_systems.append([node_ids[0], node_ids[1], node_ids[3]])
            self.shell_elements[self.last_element_key] = ShellElementClass(
                self.nodes, self.last_element_key, part_id, node_ids
            )
        return new_coord_systems

    def create_node_list_in_plane(self, plane='z', plane_loc=0.0):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        node_list_in_plane = []
        plane_map = {'x':0, 'y':1, 'z':2}
        for node in self.nodes.values():
            if node.coord[plane_map[plane.lower()]] == plane_loc:
                node_list_in_plane.append(node.id_)
        return node_list_in_plane

    def find_side_beams(self, planes=['x', 'y', 'z']): #Finding beams laying on specified planes
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        side_beams = []
        for plane in planes:
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                temp_side_beams=[]
                for beam in self.beams.values():
                    if beam.on_plane(plane, plane_loc):
                        temp_side_beams.append(beam.id_)
                side_beams.append(list(set(temp_side_beams)))
        return side_beams

    def find_side_incident_beam(self, planes=['x', 'y', 'z']): #Finding beams which have one or more nodes on specified planes
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        side_beams = []
        for plane in planes:
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                for beam in self.beams.values():
                    temp_side_beams = []
                    if beam.incident_to_plane(plane, plane_loc):
                        temp_side_beams.append(beam.id_)
                side_beams.append(list(set(temp_side_beams)))

        return side_beams

    def find_beam_node_on_side(self, planes=['x', 'y', 'z']):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        side_beam_nodes = []
        for plane in planes:
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                temp_nodes = []
                for beam in self.beams.values():
                    if beam.incident_to_plane(plane, plane_loc):
                        for elem in beam.elem_ids:
                            for node_id in self.beam_elements[elem].node_ids:
                                if self.nodes[node_id].coord[plane_map[plane]] == plane_loc:
                                    temp_nodes.append(node_id)
                side_beam_nodes.append(list(set(temp_nodes)))
        return side_beam_nodes

    def find_side_surfs(self, planes=['x', 'y', 'z']): #Finding beams laying on specified planes
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        side_surfs = []
        for plane in planes:
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                temp_side_surfs=[]
                for surf in self.surfs.values():
                    if surf.on_plane(plane, plane_loc):
                        temp_side_surfs.append(surf.id_)
                side_surfs.append(list(set(temp_side_surfs)))
        return side_surfs

    def find_side_incident_surfs(self, planes=['x', 'y', 'z']):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        plane_map = {'x': 0, 'y': 1, 'z': 2}
        side_surfs=[]
        for plane in planes:
            for plane_loc in [0.0, self.tessellation.domain_size[plane_map[plane]]]:
                temp_side_surfs=[]
                for surf in self.surfs.values():
                    if surf.incident_to_plane(plane, plane_loc):
                        temp_side_surfs.append(surf.id_)
                side_surfs.append(list(set(temp_side_surfs)))
        return side_surfs

    def find_parts_for_box_contact(self, planes=['x', 'y', 'z']):
        if self.tessellation.periodic == True: raise Exception('Invalid action for current tesselation')
        side_contact_part_list = self.find_side_incident_surfs(planes)
        side_plane_part_list = self.find_side_surfs(planes)
        volumes_on_side=[]
        for side_contact_parts, side_plane_parts in zip(side_contact_part_list, side_plane_part_list):
            volumes_on_side.append([])
            side_part_list=[int((surf_id-self.surf_num_offset)/10) for surf_id in side_contact_parts if surf_id not in side_plane_part_list]
            for volume_id, volume in zip(self.tessellation.polyhedrons.keys(), self.tessellation.polyhedrons.values()):
                if len(list(set(map(abs, volume.faces)).intersection(side_part_list)))>=2:
                    volumes_on_side[-1].append(volume_id)
        edge_parts_pr_side = [] ################# Start from here
        for volumes in volumes_on_side:
            temp_side_parts=[]
            for volume in volumes:
                mapped_surfs=list(set(abs(np.array(self.tessellation.polyhedrons[volume].faces)*10)+self.surf_num_offset))
                temp_side_parts.extend(mapped_surfs)
            edge_parts_pr_side.append(list(set(temp_side_parts).intersection(self.surfs.keys())))
        return edge_parts_pr_side

self=LSDynaGeometry(tessellation=tessellation, debug=True)