import matplotlib
import numpy as np
import copy
import scipy.optimize
import scipy.integrate
import os
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm
import math
import time
#neper -T -n 10 -id 1
class AbsDict(dict):
    def __setitem__(self, key, item):
        if isinstance(key, int):
            if np.sign(key) == -1:
                raise Exception ('Can not assign negative keys')
            else:
                self.__dict__[key] = item
        else:
            self.__dict__[key] = item

    def __getitem__(self, key):
        if isinstance(key, int):
            if np.sign(key) == -1:
                if isinstance(self.__dict__[abs(key)], list):
                    return self.__dict__[abs(key)][::-1]
                else:
                    return self.__dict__[abs(key)].reverse()
            else:
                return self.__dict__[key]
        else:
            return self.__dict__[key]


    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        if isinstance(key, int):
            del self.__dict__[abs(key)]
        else:
            del self.__dict__[key]


    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        if isinstance(k, int):
            return abs(k) in self.__dict__
        else:
            return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)
class VertexClass(object):
    def __init__(self, id_, coord, state=0):
        self.id_ = id_
        self.coord = coord
        self.state = state
        self.master_to = []
        self.slave_to = []
        self.parents = []

    def update_slave_to(self, new_master_id):
        self.slave_to[0] = new_master_id

    def update_master_to(self, old_slave_id, new_slave_id):
        target_ind = self.master_to[::4].index(old_slave_id)*4
        self.master_to[target_ind] = new_slave_id
class EdgeClass(object):
    def __init__(self, vertex_dict, id_, verts, state=0):
        self.vertex_dict=vertex_dict
        self.id_ = id_
        self.verts = verts
        self.state = state
        self.master_to = []
        self.slave_to = []
        self.parents = []

    def vector(self):
        return self.vertex_dict[self.verts[1]].coord - self.vertex_dict[self.verts[0]].coord

    def x0(self):
        return self.vertex_dict[self.verts[0]].coord

    def x1(self):
        return self.vertex_dict[self.verts[1]].coord

    def length(self):
        return np.linalg.norm(self.vector())

    def reverse(self):
        temp = EdgeClass(self.vertex_dict, id_=-self.id_, verts=self.verts[::-1], state=self.state)
        temp.master_to = self.master_to
        temp.slave_to = self.slave_to
        temp.parents = self.parents
        return temp

    def replace_vertex(self, old_id, new_id):
        if self.verts[0] == old_id:
            self.verts[0] = new_id
        elif self.verts[1] == old_id:
            self.verts[1] = new_id
        else:
            raise Exception('Could not find old vertex in edge')
class FaceClass(object):
    def __init__(self, edge_dict, id_, edges, state=0):
        self.edge_dict = edge_dict
        self.id_ = id_
        self.edges = edges
        self.state = state
        self.master_to = []
        self.slave_to = []
        self.parents= []

    def verts_in_face(self):
        return list(set([self.edge_dict[edge].verts[0] for edge in self.edges]+[self.edge_dict[edge].verts[1] for edge in self.edges]))

    def find_barycenter(self):
        return np.array([self.edge_dict[self.edges[0]].vertex_dict[vert].coord for vert in self.verts_in_face()]).mean(axis=0)

    def find_face_eq(self):
        barycenter = self.find_barycenter()
        vectors = []
        for edge in self.edges: #edgeID=self.edges[1]
            v1=self.edge_dict[edge].x0() - barycenter
            v2=self.edge_dict[edge].x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)
        averaged_vector = np.array(vectors).mean(axis=0)
        face_eq_d = np.dot(averaged_vector, barycenter)
        return [face_eq_d, averaged_vector[0], averaged_vector[1], averaged_vector[2]]

    def find_angle_deviation(self, plot_face=False):
        vectors=[]
        barycenter=self.find_barycenter()
        for edge in self.edges:
            v1=self.edge_dict[edge].x0() - barycenter
            v2=self.edge_dict[edge].x1() - barycenter
            v3 = np.cross(v1, v2)
            nv3 = v3 / np.linalg.norm(v3)
            vectors.append(nv3)

        mean_vector=np.array(vectors).mean(axis=0)
        angles=[]
        for i in range(len(vectors)):
            j = i+1
            if j ==len(vectors):
                j=0
            angles.append(np.arccos(
                np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)))

        baryangles = []
        for i in range(len(vectors)):
            baryangles.append(np.arccos(
                np.clip(np.dot(vectors[i], mean_vector), -1.0, 1.0)))
        max_angle=max(angles)
        max_angle_ind = angles.index(max_angle)
        max_bary_ind=baryangles[max_angle_ind:max_angle_ind+2].index(max(baryangles[max_angle_ind:max_angle_ind+2]))

        return [self.edges[max_angle_ind+max_bary_ind], max_angle]

    def plot_face(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(self.edges)):
            ax.plot(*np.array([self.edge_dict[self.edges[i]].x0(),self.edge_dict[self.edges[i]].x1()]).swapaxes(0,1))
        ax.scatter(*np.array([self.edge_dict[self.edges[-1]].x0(),self.edge_dict[self.edges[-1]].x1()]).swapaxes(0,1))
        ax.scatter(*np.array([self.find_barycenter()]).swapaxes(0,1))

    def remove_edge(self, old_id):
        target_ind = [abs(edge) for edge in self.edges].index(abs(old_id))
        self.edges.pop(target_ind)

    def replace_edge(self, old_id, new_id):
        replaceInd = [abs(edge) for edge in self.edges].index(abs(old_id))
        sign = np.sign(self.edges[replaceInd])
        self.edges[replaceInd] = int(sign* new_id)

    def reverse(self):
        temp = FaceClass(id_=-self.id_, edges=[-1 * edge for edge in self.edges[::-1]], state = self.state)
        temp.master_to = self.master_to
        temp.slave_to  = self.slave_to
        temp.parents = self.parents
        return temp
class PolyhedronClass(object):
    def __init__(self, face_dict, id_, faces):
        self.face_dict = face_dict
        self.id_ = id_
        self.faces = faces

    def removeFace(self, old_id):
        target_ind = [abs(face) for face in self.faces].index(abs(old_id))
        self.faces.pop(target_ind)
class Tessellation(object):
    '''Provide path and name of .tess file created with Neper'''
    def __init__(self, tess_file_name, mesh_file_name=None):
        self.tess_file_name=tess_file_name
        self.mesh_file_name= mesh_file_name
        with open(self.tess_file_name, 'r') as tess_raw:
            self.lines=tess_raw.readlines()

        self.vertices = self.get_vertices()
        self.edges = self.get_edges()
        self.faces = self.get_faces()
        self.polyhedrons = self.get_polyhedrons()
        self.gmsh = []
        self.rejected_edge_del = []
        self.edge_lengths = self.find_edge_lengths()
        self.domain_size = self.get_domain_size()
        self.find_parents()
        self.periodic = False
        if  ' **periodicity\n' in self.lines:
            self.periodic = True
            self.find_parents()
            self.get_periodicity()
             # For storing  rejected edges, duch that they are not tried again.
            self.vertex_id_counter = max(self.vertices.keys())
            self.edge_id_counter = max(self.edges.keys())


    def get_vertices(self):
        vertices={}
        start_ind=self.lines.index(' **vertex\n')
        for line in self.lines[start_ind+2:start_ind+2+int(self.lines[start_ind+1])]:
            id_ = int(line.split()[0])
            coord = np.array(list(map(float, line.split()[1:-1])))
            vertices[id_] = VertexClass(id_=id_, coord=coord)
        return vertices

    def get_edges(self):
        edges=AbsDict()
        start_ind=self.lines.index(' **edge\n')
        for line in self.lines[start_ind+2:start_ind+2+int(self.lines[start_ind+1])]:
            id_ = int(line.split()[0])
            verts = list(map(int, line.split()[1:3])) #Edge vertex 0 and 1
            edges[id_] = EdgeClass(self.vertices, id_=id_, verts=verts)
        return edges

    def get_faces(self):
        faces = AbsDict()
        start_ind = self.lines.index(' **face\n')
        num_faces = int(self.lines[start_ind+1])
        for i in range(num_faces):
            vertex_line_ind = start_ind + 2 + i*4
            edge_line_ind = vertex_line_ind + 1
            face_edges = list(map(int, self.lines[edge_line_ind].split()[1:]))
            id_=int(self.lines[vertex_line_ind].split()[0])
            faces[id_] = FaceClass(self.edges, id_=id_, edges=face_edges)
        return faces

    def get_polyhedrons(self):
        polyhedrons = {}
        start_ind = self.lines.index(' **polyhedron\n')
        n_polyhedrons = int(self.lines[start_ind + 1])
        for i in range(n_polyhedrons):
            polyhedron_line_ind = start_ind+2+i
            id_ = int(self.lines[polyhedron_line_ind].split()[0])
            poly_faces = list(map(int, self.lines[polyhedron_line_ind].split()[2:]))
            polyhedrons[id_] = PolyhedronClass(self.faces, id_=id_, faces=poly_faces)
        return polyhedrons

    def get_domain_size(self):
        start_ind = self.lines.index(' **domain\n')
        # domain_size = np.array(list(map(float, self.lines[domain_size_ind].split())))

        domain_start_ind = start_ind + 5
        n_verts = 8
        domain = {}
        for line in self.lines[domain_start_ind: domain_start_ind + n_verts*2:2]: #line=self.lines[domain_start_ind: domain_start_ind + n_verts*2:2] [0]
            id_ = int(line.split()[0])
            coord = np.array(list(map(float, line.split()[1:-1])))
            domain[id_] = coord
        return domain[7]-domain[1]

    def get_periodicity(self):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        periodicity_start_ind = self.lines.index(' **periodicity\n')
        vertex_start_ind = periodicity_start_ind+ self.lines[periodicity_start_ind:].index('  *vertex\n')
        n_verts = int(self.lines[vertex_start_ind+1])
        for line in self.lines[vertex_start_ind+2: vertex_start_ind+2+n_verts]:
            id_0 = int(line.split()[0])
            id_1 = int(line.split()[1])
            self.vertices[id_0].slave_to.extend(list(map(int, line.split()[1:])))
            self.vertices[id_1].master_to.extend([id_0] + list(map(int, line.split()[2:])))

        edge_start_ind = periodicity_start_ind + self.lines[periodicity_start_ind:].index('  *edge\n')
        n_edges = int(self.lines[edge_start_ind+1])
        for line in  self.lines[edge_start_ind+2: edge_start_ind+2+n_edges]:
            id_0 = int(line.split()[0])
            id_1 = int(line.split()[1])
            self.edges[id_0].slave_to.extend(list(map(int, line.split()[1:])))
            self.edges[id_1].master_to.extend([id_0] + list(map(int,line.split()[2:])))

        face_start_ind = periodicity_start_ind + self.lines[periodicity_start_ind:].index('  *face\n')
        n_faces = int(self.lines[face_start_ind + 1])
        for line in self.lines[face_start_ind + 2: face_start_ind + 2 + n_faces]:
            id_0 = int(line.split()[0])
            id_1 = int(line.split()[1])
            self.faces[id_0].slave_to.extend(list(map(int, line.split()[1:])))
            self.faces[id_1].master_to.extend([id_0] + list(map(int,line.split()[2:])))

    def check_if_periodic(self, master_coord, slave_coord):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        coord_offset = slave_coord - master_coord
        offset_is_zero = [math.isclose(offset, 0.0, rel_tol=1e-8, abs_tol=0.0) for offset in coord_offset]
        offset_as_unity = np.array(list(map(int,[not i for i in offset_is_zero])))
        comping_coord = slave_coord + (offset_as_unity * self.domain_size * -1*np.sign(coord_offset))
        if self.compare_arrays(master_coord, comping_coord) == True:
            return coord_offset
        else:
            return np.array([None, None, None])

    def get_periodicity_internal_update(self, affected_vertices):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        affected_edges = []
        for vertex in affected_vertices:
            self.vertices[vertex].master_to = []
            self.vertices[vertex].slave_to = []
            affected_edges.extend(self.vertices[vertex].parents)

        for edge in affected_edges:
            self.edges[edge].master_to = []
            self.edges[edge].slave_to = []

        t = time.time()
        checked_vertices=[]
        for vertex in affected_vertices: #vertex = affected_vertices[0]
            for slave_vertex in affected_vertices:#slave_vertex = affected_vertices[0]
                if vertex != slave_vertex and slave_vertex not in checked_vertices and vertex not in checked_vertices:
                    master_coord = self.vertices[vertex].coord
                    slave_coord = self.vertices[slave_vertex].coord
                    coord_offset = self.check_if_periodic(master_coord, slave_coord)
                    if all(coord_offset != [None, None, None]):
                        self.vertices[vertex].master_to.extend(
                            [slave_vertex] + [int(np.sign(round(val, 3))) for val in coord_offset])
                        self.vertices[slave_vertex].slave_to.extend(
                            [vertex] + [int(np.sign(round(val, 3))) for val in coord_offset])
                        checked_vertices.append(slave_vertex)
            checked_vertices.append(vertex)

        elapsed = time.time() - t
        #print('Time to find vertex periodicity: {:.3f} s'.format(elapsed))

        checked_edge_list = []
        for edge in affected_edges: #edge =  affected_edges[0]
            edge = self.edges[edge]
            if edge.id_ not in checked_edge_list:
                master_verts = edge.verts
                connected_verts = []
                for master_vert in master_verts:
                    if self.vertices[master_vert].master_to != []:
                        connected_verts.extend(self.vertices[master_vert].master_to[::4])
                    elif self.vertices[master_vert].slave_to != []:
                        connected_verts.extend(self.vertices[self.vertices[master_vert].slave_to[0]].master_to[::4])
                        connected_verts.remove(master_vert)
                parent_edges = set([parent_edge for connected_vert in connected_verts for parent_edge in
                                    self.vertices[connected_vert].parents])
                master_vector = edge.vector()

                for parent_edge in parent_edges: #parentEdgeID = 51
                    if self.compare_arrays(self.edges[parent_edge].vector(), master_vector):
                        coord_offset = self.edges[parent_edge].x0()-edge.x0()
                        self.edges[edge.id_].master_to.extend(
                            [parent_edge] + [int(np.sign(round(val, 3))) for val in coord_offset] +[1]
                        )
                        self.edges[parent_edge].slave_to.extend(
                            [edge.id_] + [int(np.sign(round(val, 3))) for val in coord_offset] + [1]
                        )
                        checked_edge_list.append(parent_edge)

                    elif self.compare_arrays(-1*self.edges[parent_edge].vector(), master_vector):
                        periodicity = self.edges[parent_edge].x0()-edge.x1()
                        self.edges[edge.id_].master_to.extend(
                            [parent_edge] + [int(np.sign(round(val, 3))) for val in periodicity] + [-1]
                        )
                        self.edges[parent_edge].slave_to.extend(
                            [edge.id_] + [int(np.sign(round(val, 3))) for val in periodicity] + [-1]
                        )
                        checked_edge_list.append(parent_edge)
            checked_edge_list.append(edge.id_)

    def find_parents(self):
        for vertex_key in self.vertices.keys():
            self.vertices[vertex_key].parents = []

        for edge_key in self.edges.keys():
            self.edges[edge_key].parents = []

        for face_key in self.faces.keys():
            self.faces[face_key].parents = []

        for edge in self.edges.values():
            for ver in edge.verts:
                if edge.id_ not in self.vertices[ver].parents:
                    self.vertices[ver].parents.append(edge.id_)

        for face in self.faces.values():
            for edge in face.edges:
                if face.id_ not in self.edges[abs(edge)].parents:
                    self.edges[abs(edge)].parents.append(face.id_)

        for poly in self.polyhedrons.values():
            for face in poly.faces:
                if poly.id_ not in self.faces[abs(face)].parents:
                    self.faces[abs(face)].parents.append(poly.id_)

    def find_edge_lengths(self):
        lengths =  np.array([[edge.length(), int(edge.id_)] for edge in self.edges.values()
                             if edge.id_ not in self.rejected_edge_del])
        if len(lengths) == 0:
            print('No more edges to find')
            return []
        else:
            lengths = lengths[lengths[:, 0].argsort()]
            return lengths

    def find_new_vertices(self, edges, edge_periodicities, vertices, vertex_periodicities):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        def distance_to_plane(point, plane_equation):
            # planeEquation[1:] should be unity
            return abs(np.dot(plane_equation[1:], point) - plane_equation[0])

        def lsq_distance(point, plane_equations):
            return np.sqrt(sum([distance_to_plane(point + plane_equation[1], np.array(plane_equation[0])) ** 2
                 for plane_equation in plane_equations]))

        plane_equations=[]
        starting_point = np.array([edges[0].x0(), edges[0].x1()]).mean(axis=0)
        for edge, periodicity in zip(edges, edge_periodicities):
            vertex_edges = [edge_id for vert_id in edge.verts for edge_id in
                           self.vertices[vert_id].parents]
            connected_faces = [self.faces[face_id].id_ for edge_id in vertex_edges for
                              face_id in self.edges[edge_id].parents]
            plane_equations.extend([[self.faces[face_id].find_face_eq()] + [periodicity*self.domain_size] for face_id in
                              set(connected_faces)])

        for vertex, periodicity in zip(vertices, vertex_periodicities):
            vertex_edges = [edge_id for edge_id in
                           self.vertices[vertex.id_].parents]
            connected_faces = [self.faces[face_id].id_ for edge_id in vertex_edges for
                              face_id in self.edges[edge_id].parents]
            plane_equations.extend([[self.faces[face_id].find_face_eq()] + [periodicity*self.domain_size] for face_id in
                                    set(connected_faces)])

        new_master_vertex=scipy.optimize.minimize(lsq_distance, starting_point, plane_equations,).x
        new_edge_vertices = [new_master_vertex+periodicity*self.domain_size for periodicity in edge_periodicities]
        new_vertex_vertices = [new_master_vertex+periodicity*self.domain_size for periodicity in vertex_periodicities]
        return new_edge_vertices, new_vertex_vertices

    def find_periodic_dependecies(self, edge):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        ############################################################3
        #First, edge dependecies should be found
        #########################################################
        if edge.slave_to == [] and edge.master_to == []: # Not slave or master
            master_edge=edge
        if edge.slave_to != []:
            # If slave to is not empty, this edge is a slave.
            # The only master for this slave edge is edge.slave_to[0]
            master_edge_id = edge.slave_to[0]
            #The new master edge is now assigned
            master_edge = self.edges[master_edge_id]
        elif edge.master_to != []:
            # If master to is not empty, this edge can not be a slave
            # The master edge is now assigned as itself.
            master_edge=edge
        # the dependency list is initiated with the master edge_id
        dep_edge_ids = [master_edge.id_]
        #Each edge id subject to the master edge is added.
        for i in range(0, len(master_edge.master_to), 5):
            dep_edge_ids.append(master_edge.master_to[i])
        ############################################################3
        # Then, edges and corresponding periodicities should be found
        #########################################################
        #Initializing with master edge
        edge_periodicities = [np.array([0, 0, 0])]
        edges = [self.edges[dep_edge_ids[0]]]
        for id_ in dep_edge_ids[1:]: #dep_edge_id = dep_edge_ids[1]
            edge_periodicities.append(np.array(self.edges[id_].slave_to[1:4]))
            edges.append(self.edges[id_])
        ############################################################3
        #Now vertex dependecies should be found
        #########################################################
        verts = [*self.edges[dep_edge_ids[0]].verts]
        vert_periodicities = [np.array([0, 0, 0]), np.array([0, 0, 0])]
        # The slave edge vertices are added to the list with their respective periodicities, relative to the master edge vertices
        for dep_edge_id in dep_edge_ids[1:]:
            verts.extend(self.edges[dep_edge_id].verts)
            vert_periodicities.extend(
                [np.array(self.edges[dep_edge_id].slave_to[1:4]), np.array(self.edges[dep_edge_id].slave_to[1:4])])
        #The edge vertices are recorded
        edge_verts = copy.copy(verts)
        #The edge vertices are checked for periodicities outside the edge periodicities
        for id_, periodicity in zip(verts, vert_periodicities):  # ver, periodicity = verts[0], vert_periodicities[0]
            if self.vertices[id_].slave_to != []:
                if self.vertices[id_].slave_to[0] not in verts:
                    verts.append(self.vertices[id_].slave_to[0])
                    vert_periodicities.append(periodicity - np.array(self.vertices[id_].slave_to[1:4]))
            elif self.vertices[id_].master_to != []:
                for i in range(0, len(self.vertices[id_].master_to), 4):
                    if self.vertices[id_].master_to[i] not in verts:
                        verts.append(self.vertices[id_].master_to[i])
                        vert_periodicities.append(np.array(periodicity + self.vertices[id_].master_to[i + 1:i + 4]))
        # The collected vertices are then filtered to remove the edge vertices
        vertices = [self.vertices[vertex] for vertex, periodicity in zip(verts, vert_periodicities) if
                    vertex not in edge_verts]
        vertex_periodicities = [periodicity for vertex, periodicity in zip(verts, vert_periodicities) if
                               vertex not in edge_verts]

        return edges, edge_periodicities, vertices, vertex_periodicities

    def remove_edge(self, edge_id, del_layer = 0):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        print_trigger = False
        if del_layer == 0: print_trigger=True
        t = time.time()
        edge=self.edges[edge_id]
        # The new vertices are found as offsets from the master edge.
        edges, edge_periodicities, vertices, vertex_periodicities = self.find_periodic_dependecies(edge)

        #The new vertex location for a edge collapse, and new vertex locations for moved slave vertices are found
        new_edge_vertices, new_vertex_vertices= self.find_new_vertices(edges, edge_periodicities, vertices, vertex_periodicities)

        new_vertex_list=[]
        collapsed_faces_list=[]
        deleted_verts_list=[]
        coalesced_edges = []

        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to find dependencies: {:.3f} s'.format(elapsed))

        #For each edge and corresponding vertex location, the edge is merged to a single point.
        #The maximim deviation of each affected face is returned.
        t = time.time()

        for edge, new_edge_vertex in zip(edges, new_edge_vertices): #edge, newEdgeVertex = list(zip(edges, new_edge_vertices))[0]
            new_vertex_id, collapsed_faces, deleted_vertices = self.replace_edge_with_vertex(edge, new_edge_vertex, print_trigger)
            new_vertex_list.append(new_vertex_id)
            collapsed_faces_list.append(collapsed_faces)
            deleted_verts_list.extend(deleted_vertices)


        for vertex, new_vertex_loc in zip(vertices, new_vertex_vertices): #vertex, newVertexLoc = list(zip(vertices, new_vertex_vertices))[0]
            updated_vertex = self.update_vertex_location(vertex, new_vertex_loc)
            new_vertex_list.append(updated_vertex)
            collapsed_faces_list.append([])
            deleted_verts_list.append([])

        elapsed = time.time() - t
        if del_layer == 0:
            print('Time to delete dependencies: {:.3f} s'.format(elapsed))

        t = time.time()
        #Check if dependent vertices have merged:
        duplicate_vertex_sets = self.check_for_duplicate_vertices(vertex_list=vertices)
        #print('Time to check for duplicate vertices: {:.3f} s'.format(elapsed))

        t = time.time()
        if duplicate_vertex_sets != []:
            for duplicate_vertex_set in duplicate_vertex_sets: #duplicateVertexSet = duplicateVertexSets[0]
                new_vertex_list.append(self.resolve_duplicate_vertices(duplicate_vertex_set))
                collapsed_faces_list.append([])
                deleted_verts_list.extend(duplicate_vertex_set)
                #Check if edges have merged #######
                parents=self.vertices[new_vertex_list[-1]].parents
                duplicate_edge_sets = self.check_for_duplicate_edges(edge_list=[self.edges[parent_edge_id] for parent_edge_id in parents])
                if duplicate_edge_sets != []:
                    for duplicate_edge_set in duplicate_edge_sets: #duplicateEdgeSet = duplicateEdgeSets[0]
                        coalesced_edges.append(self.resolve_duplicate_edges(duplicate_edge_set)) # coalescedEdges.append(newEdgeID)

        for collapsed_faces_pr_edge in collapsed_faces_list:
            if collapsed_faces_pr_edge != []:
                for collapsed_face in collapsed_faces_pr_edge: #collapsedFace=820
                    coalesced_edges.append(self.delete_face_to_edge(collapsed_face, print_trigger))

        for vertex in deleted_verts_list:
            if vertex in new_vertex_list:
                new_vertex_list.remove(vertex)

        for vert in new_vertex_list:  # vert = new_vertex_list[2]
            parents = self.vertices[vert].parents
            duplicate_edge_sets = self.check_for_duplicate_edges(
                edge_list=[self.edges[parent_edge_id] for parent_edge_id in parents if
                           parent_edge_id in self.edges.keys()])
            if duplicate_edge_sets != []:
                for duplicate_edge_set in duplicate_edge_sets:  # duplicateEdgeSet = duplicateEdgeSets[0]
                    coalesced_edges.append(
                        self.resolve_duplicate_edges(duplicate_edge_set))  # coalescedEdges.append(newEdgeID)

        elapsed = time.time() - t
       # print('Time to deal with duplicate vertices: {:.3f} s'.format(elapsed))
        affected_vertices = copy.copy(new_vertex_list)

        for edge_id in coalesced_edges: #edge = coalesced_edges[0]
            try:  # edge deleted in double face deletion deletion
               edge = self.edges[edge_id]
               for dep_edge in self.find_periodic_dependecies(edge)[0]:
                   affected_vertices.extend(dep_edge.verts)
               for dep_vert in self.find_periodic_dependecies(edge)[2]:
                   affected_vertices.append(dep_vert.id_)
            except:
                print('Coalleced edge might have been deleted')
                pass


        #Update the vertex and edge periodicity of the affected edges
        self.find_parents()
        self.get_periodicity_internal_update(affected_vertices)
        #self.findParents()
        ####################################################################
        #Find all affected edges, by newVertexList, and check internal angles
        ########################################################################
        affected_edges = [edge_id for vert_id in new_vertex_list for edge_id in self.vertices[vert_id].parents]
        angles = np.array([self.faces[face_id].find_angle_deviation()
                           for edge_id in affected_edges for face_id in self.edges[edge_id].parents])
        sorted_angles = angles[angles[:,1].argsort()[::-1],:]
        self.new_vertex_list = new_vertex_list
        self.deleted_verts_list = deleted_verts_list
        if sorted_angles[0,1] < 20 * np.pi / 180.:
            return True
        elif del_layer == 0:
            checked_edges = []
            for edge_angle in sorted_angles: #edge_angle =  sorted_angles[0]
                if edge_angle[1] > 20 * np.pi / 180. and int(abs(edge_angle[0])) not in checked_edges:
                    try:
                        layer_edge_id = int(abs(edge_angle[0]))
                        dep_edges = self.find_periodic_dependecies(self.edges[layer_edge_id])[0]
                        checked_edges.extend(dep_edge.id_ for dep_edge in dep_edges)
                        self.tess_copy = copy.deepcopy(self)
                        #new_vertex_list_layer = self.tess_copy.remove_edge(layer_edge_id, del_layer=del_layer + 1)
                        if self.tess_copy.remove_edge(layer_edge_id, del_layer=del_layer + 1):
                            new_vertex_list.extend(self.tess_copy.new_vertex_list)
                            for vert in self.tess_copy.deleted_verts_list:
                                if vert in new_vertex_list:
                                    new_vertex_list.remove(vert)
                            self.vertices = self.tess_copy.vertices
                            self.edges = self.tess_copy.edges
                            self.faces = self.tess_copy.faces
                            self.polyhedrons = self.tess_copy.polyhedrons
                            self.edge_lengths = self.tess_copy.edge_lengths
                            self.vertex_id_counter = self.tess_copy.vertex_id_counter
                            self.edge_id_counter= self.tess_copy.edge_id_counter
                            print ('{} st/nd layer deletion of edge {} was successful'.format(
                                del_layer+1, int(edge_angle[0])))
                            print('--------------------------------------------------------------')
                        else:
                            self.tess_copy = []
                            print('{} st/nd layer deletion of edge {} failed with new angle {}'.format(
                                del_layer+1, int(edge_angle[0]), edge_angle[1]))
                    except:
                        print ('Error encountered in {} st/nd layer deletion of edge {}'.format(
                                del_layer+1, int(edge_angle[0])))
                else:
                    pass
            new_vertex_list_final=set(new_vertex_list)
            filtered_vertex_list=[vert_id for vert_id in new_vertex_list_final if vert_id in self.vertices.keys()]
            affected_edges = [edge_id for vert_id in filtered_vertex_list for edge_id in self.vertices[vert_id].parents
                              if edge_id in self.edges.keys()]
            angles = np.array([self.faces[face_id].find_angle_deviation()
                               for edge_id in affected_edges for face_id in self.edges[edge_id].parents])
            sorted_angles = angles[angles[:, 1].argsort()[::-1], :]
            if sorted_angles[0, 1] < 20 * np.pi / 180.:
                return True
            else:
                return False
        else:
            return False

    def delete_face_to_edge(self, collapsed_face, print_trigger=False):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        # if the collapsed face does not belong in a slave/master combo, but the deleted edge did, the edge to be deleted should not move the masterVertex.
        #[self.tess_copy.edges[edge].master_to for edge in self.tess_copy.faces[39].edges]
        rem_edges = self.faces[collapsed_face].edges
        # Remove face from poly parent
        for poly_parent in self.faces[collapsed_face].parents:
            self.polyhedrons[poly_parent].removeFace(collapsed_face)

        # Merge the two edges for all remaining faces
        self.edge_id_counter += 1
        new_edge_id = self.edge_id_counter
        state = max([self.edges[edge_id].state for edge_id in rem_edges])

        # Update vertex parent relations before assigning
        new_edge_vertices = self.edges[rem_edges[0]].verts
        new_edge = EdgeClass(self.vertices, id_=new_edge_id, verts=new_edge_vertices, state=state + 1)
        new_edge.parents = list(set([parent_face_id for remEdge in rem_edges for parent_face_id in
                                    self.edges[remEdge].parents if parent_face_id != collapsed_face]))


        # for each old edge, remove and replace with new edge
        for old_edge in rem_edges: #oldEdge=remEdges[0]
            # Find all parent faces and replace
            for edge_rem_face_id in self.edges[old_edge].parents:
                if edge_rem_face_id != collapsed_face:
                    if all(self.edges[abs(old_edge)].vector() == new_edge.vector()):
                        self.faces[edge_rem_face_id].replace_edge(old_edge, new_edge_id)
                    elif all(self.edges[abs(old_edge)].vector() == new_edge.reverse().vector()):
                        self.faces[edge_rem_face_id].replace_edge(old_edge, -1 * new_edge_id)
                    else:
                        raise Exception('Not the same vector of edges being merged. Face_id {}'.format(collapsed_face))

        self.edges[new_edge_id] = new_edge
        if print_trigger == True:
            print('Suggested face for deletion: face {}'.format(collapsed_face))
            print('Coalesced edges {},{} to edge: {}'.format(abs(rem_edges[0]), abs(rem_edges[1]), new_edge_id))

        del self.faces[abs(collapsed_face)]
        del self.edges[abs(rem_edges[0])]
        del self.edges[abs(rem_edges[1])]
        return new_edge_id

    def update_vertex_location(self, vertex, new_vertex_loc):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        self.vertices[vertex.id_].state+=1
        self.vertices[vertex.id_].coord = new_vertex_loc
        return vertex.id_

    def replace_edge_with_vertex(self, edge, new_edge_vertex, print_trigger=False):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        # The maximum number of edits from the two vertexs' is recovered
        vert_state = max([self.vertices[vert].state for vert in edge.verts])
        #The list of vertices  to be removed is found
        old_vert_list=edge.verts
        #Find new vertex_id from the maximum value in the list  +1
        self.vertex_id_counter += 1
        new_vertex_id = self.vertex_id_counter

        #Create the new vertex with the new coordinate and ver_state = ver_state +1
        self.vertices[new_vertex_id] = VertexClass(id_=new_vertex_id, coord = new_edge_vertex, state=vert_state + 1)

        # Initiate list of all edges about to be affected by the merging
        affected_edges = []
        collapsed_faces = []
        # For each vertex about to be merged, update all affected edges with new vertex_id
        for vert_id in old_vert_list: #ver_id = old_ver_list[0]
            #Find edges connected to each vertices about to be merges. Exlude the edge to be removed
            affected_edges_pr_vert = list(set([parent_edge_id for parent_edge_id in self.vertices[vert_id].parents if parent_edge_id != edge.id_]))
            for affected_edge in affected_edges_pr_vert:
                #Find index in edge.verts where old vertex is located
                self.edges[affected_edge].replace_vertex(vert_id, new_vertex_id)
            # Add edges from each vertex to the collection
            affected_edges.extend(affected_edges_pr_vert)

        #Assign vertex parent list to new vertex. All affected edges are parents.
        self.vertices[new_vertex_id].parents = affected_edges

        #Remove deleted edge from affected faces
        edge_faces = self.edges[edge.id_].parents
        for face_id in edge_faces:
            self.faces[face_id].remove_edge(edge.id_)
            #Check if face has collapsed:

            # If face eliminated:
            if len(self.faces[face_id].edges) <= 2:
                collapsed_faces.append(abs(face_id))
        if print_trigger == True:
            print ('Suggested edge for deletion: edge {}'.format(edge.id_))
            print ('New vertex ID: {}'.format(new_vertex_id))
        del self.edges[edge.id_]
        for vert in old_vert_list:
            del self.vertices[vert]
        #self.edge_lengths = self.findEdgeLengths()
        return new_vertex_id, collapsed_faces, old_vert_list

    def resolve_duplicate_vertices(self, duplicate_vertices):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        self.vertex_id_counter += 1
        new_vertex_id = self.vertex_id_counter
        state = max([self.vertices[vert].state for vert in duplicate_vertices])
        # Create the new vertex with the new coordinate and ver_state = ver_state +1
        parent_list=[]
        for dup_id in duplicate_vertices:
            for edge in self.vertices[dup_id].parents:
                parent_list.append(edge)
                self.edges[edge].replace_vertex(dup_id, new_vertex_id)
        self.vertices[new_vertex_id] = VertexClass(id_=new_vertex_id, coord=self.vertices[dup_id].coord, state=state + 1)
        self.vertices[new_vertex_id].parents = copy.copy(parent_list)
        for dup_id in duplicate_vertices:
            del self.vertices[dup_id]
        return new_vertex_id

    def resolve_duplicate_edges(self, duplicate_edges):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        rem_edges = duplicate_edges
        # Merge the two edges for all remaining faces
        self.edge_id_counter += 1
        new_edge_id = self.edge_id_counter
        state = max([self.edges[edge_id].state for edge_id in rem_edges])

        # Update vertex parent relations before assigning
        new_edge_vertices = self.edges[rem_edges[0]].verts
        new_edge = EdgeClass(self.vertices, id_=new_edge_id, verts=new_edge_vertices, state=state + 1)
        new_edge.parents = list(set([parent_face_id for rem_edge in rem_edges for parent_face_id in
                                    self.edges[rem_edge].parents]))

        # for each old edge, remove and replace with new edge
        for old_edge in rem_edges:  # oldEdge=remEdges[0]
            # Find all parent faces and replace
            for edge_rem_face_id in self.edges[old_edge].parents:
                if all(self.edges[abs(old_edge)].vector() == new_edge.vector()):
                    self.faces[edge_rem_face_id].replace_edge(old_edge, new_edge_id)
                elif all(self.edges[abs(old_edge)].vector() == new_edge.reverse().vector()):
                    self.faces[edge_rem_face_id].replace_edge(old_edge, -1 * new_edge_id)
                else:
                    raise Exception('Not the same vector of edges being merged. Edge ID: {}, {}'.format(*duplicate_edges))

        self.edges[new_edge_id] = new_edge
        del self.edges[abs(rem_edges[0])]
        del self.edges[abs(rem_edges[1])]
        return new_edge_id

    def evaluate_remove_edge(self, edge_id):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        #The current structure is copied and all operations happen on this copy.
        self.tess_copy = copy.deepcopy(self)
        #The edge is deleted in the copy, returning the new vertex ids and angle deviation.
        if self.tess_copy.remove_edge(edge_id):
            self.vertices = self.tess_copy.vertices
            self.edges = self.tess_copy.edges
            self.faces = self.tess_copy.faces
            self.polyhedrons = self.tess_copy.polyhedrons
            self.edge_lengths = self.find_edge_lengths()
            self.vertex_id_counter = self.tess_copy.vertex_id_counter
            self.edge_id_counter = self.tess_copy.edge_id_counter
            print('Delete accepted, structure updated')
            print('----------------------------------------')
            self.tess_copy=[]

        else:
            print ('Delete of edge {} rejected'.format(edge_id))
            print('----------------------------------------')
            edge = self.edges[edge_id]
            edges, edge_periodicities, vertices, vertex_periodicities = self.find_periodic_dependecies(edge)
            for similar_edge in edges:
                self.rejected_edge_del.append(similar_edge.id_)
            self.edge_lengths = self.find_edge_lengths()
            self.tess_copy = []

    def regularize(self, n):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        for i in range(n):
            print(i)
            if len(self.edge_lengths) < 2:
                print('No more edges to check!')
                break
            edge_id = int(self.edge_lengths[0, 1])
            self.evaluate_remove_edge(edge_id)
            #if i%1 == 0:
            self.check_periodicity_face()
            if i % 1 == 0:
                if self.check_for_duplicate_vertices() != []:
                    raise Exception('Duplicate vertices happened: {}'.format(i))
                if self.check_for_duplicate_edges() != []:
                    #duplicate_edges=self.check_for_duplicate_edges()
                    #for duplicate_edge in duplicate_edges:
                        #self.resolve_duplicate_edges(duplicate_edge)
                    #print ('Duplicate edges happened: {}'.format(i))
                    raise Exception('Duplicate edges happened: {}'.format(i))

    def check_for_duplicate_vertices(self, vertex_list=[]):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        vertex_loc_list = []
        duplicate_coord_set = []
        if vertex_list == []:
            vertex_list = list(self.vertices.values())
        for i, vertex in enumerate(vertex_list):
            if list(vertex.coord) in vertex_loc_list:
                duplicate_coord_set.append([vertex.id_, vertex_list[vertex_loc_list.index(list(vertex.coord))].id_])
            vertex_loc_list.append(list(vertex.coord))
        return duplicate_coord_set

    def check_for_duplicate_edges(self, edge_list=[]):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        edge_loc_list = []
        duplicate_edge_set = []
        if edge_list == []:
            edge_list = list(self.edges.values())
        for edge in edge_list:
            edge_center_coord = (edge.x0() + edge.x1()) / 2.
            if list(edge_center_coord) in edge_loc_list:
                duplicate_edge_set.append([edge.id_, edge_list[edge_loc_list.index(list(edge_center_coord))].id_])
            edge_loc_list.append(list(edge_center_coord))
        return duplicate_edge_set

    def check_periodicity_vertex(self):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        for vert in self.vertices.values():
            if vert.master_to != []:
                for i in range(0, len(vert.master_to), 4):
                    slave_period = vert.master_to[i:i + 4]
                    master_coord = vert.coord
                    slave_coord = self.vertices[slave_period[0]].coord
                    comping_coord = slave_coord - np.array(slave_period[1:4])*self.domain_size
                    if self.compare_arrays(master_coord, comping_coord) == False:
                        print('Master vertex {} and slave vertex {} no longer periodic'.format(vert.id_, slave_period[0]))

    def check_periodicity_edge(self):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        for edge in self.edges.values():
            if edge.master_to != []:
                for i in range(0, len(edge.master_to), 5):
                    slave_period = edge.master_to[i:i + 5]
                    master_coord = edge.vector()
                    slave_coord = self.edges[slave_period[0]].vector()
                    comping_coord = slave_coord
                    if self.compare_arrays(master_coord, comping_coord) == False:
                        print('Master edge {} and slave edge {} no longer periodic'.format(edge.id_, slave_period[0]))
                        raise Exception('Faces not periodic'.format(i))

    def check_periodicity_face(self):
        if self.periodic == False: raise Exception('Invalid action for current tesselation')
        for face in self.faces.values():
            if face.master_to != []:
                for i in range(0, len(face.master_to), 5):
                    slave_period = face.master_to[i:i+5]
                    master_coord = face.find_barycenter()
                    slave_coord = self.faces[slave_period[0]].find_barycenter()
                    comping_coord = slave_coord - np.array(slave_period[1:4])*self.domain_size
                    if self.compare_arrays(master_coord, comping_coord) == False:
                        print ('Master face {} and slave face {} no longer periodic'.format(face.id_, slave_period[0]))
                        raise Exception('Faces not periodic'.format(i))
        print ('All faces still periodic')

    def compare_arrays(self, arr0, arr1, rel_tol=1e-07, abs_tol=0.0):
        return all([math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(arr0, arr1)])

    def outer_faces(self):
        outer_faces = []
        for face in self.faces.values():
            if face.master_to != []:
                outer_faces.append(face.id_)
                outer_faces.extend(face.master_to[::5])
        return outer_faces

    def plot_faces(self, faces=[]):
        fig = plt.figure()
        if faces==[]:
            all_faces = list(self.faces.keys())
            faces=all_faces
            ax = Axes3D(fig)
            colors=cm.tab20b.colors
            len_color_map=len(colors) #Accent
            color_map={}
            i=0
            for face_id in all_faces:
                if face_id not in color_map.keys():
                    color_map[face_id] = colors[i]
                if i==len_color_map-1:
                    i=0
                else:
                    i+=1
            for face_id in faces:
                if self.faces[face_id].slave_to!=[]:
                    color_map[face_id] = color_map[self.faces[face_id].slave_to[0]]



            for face_id in faces:
                coord = np.array([self.edges[edge_id].x0() for edge_id in self.faces[face_id].edges]).swapaxes(0, 1)
                X = list(coord[0])
                Y = list(coord[1])
                Z = list(coord[2])
                vertices = [list(zip(X, Y, Z))]
                face = Poly3DCollection(vertices, linewidths=1)
                edge = Line3DCollection(vertices)
                face.set_facecolor(color_map[face_id])
                edge.set_edgecolor('k')
                ax.add_collection3d(face)
                ax.add_collection3d(edge)
               # plt.show()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        else:
            ax = Axes3D(fig)
            all_faces = list(self.faces.keys())
            colors = cm.tab20b.colors
            len_color_map = len(colors)  # Accent
            color_map = {}
            i = 0
            for face_id in all_faces:
                if face_id not in color_map.keys():
                    color_map[face_id] = colors[i]
                if i == len_color_map - 1:
                    i = 0
                else:
                    i += 1
            for face_id in faces:
                if self.faces[face_id].slave_to != []:
                    color_map[face_id] = color_map[self.faces[face_id].slave_to[0]]
            for face_id in faces:
                coord = np.array([self.edges[edgeID].x0() for edgeID in self.faces[face_id].edges]).swapaxes(0, 1)
                X = list(coord[0])
                Y = list(coord[1])
                Z = list(coord[2])
                vertices = [list(zip(X, Y, Z))]
                face = Poly3DCollection(vertices, linewidths=1)
                edge = Line3DCollection(vertices)
                face.set_facecolor(color_map[face_id])
                edge.set_edgecolor('k')
                ax.add_collection3d(face)
                ax.add_collection3d(edge)
                # plt.show()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def write_geo(self, mesh_file_name = None):
        if self.mesh_file_name==None and mesh_file_name==None:
            self.mesh_file_name=self.tess_file_name
        elif self.mesh_file_name == None and mesh_file_name != None:
            self.mesh_file_name=mesh_file_name
        else:
            pass
        with open(self.mesh_file_name.rsplit('.')[0] + '.geo', 'w+') as geo_file:
            geo_file.write('SetFactory("OpenCASCADE");\n')
            for id, vert in zip(self.vertices.keys(), self.vertices.values()):
                geo_file.write('Point ({id}) = {{{:.10f}, {:.10f}, {:.10f}}};\n'.format(id=id, *vert.coord))

            for id, edge in zip(self.edges.keys(), self.edges.values()):
                geo_file.write('Line ({id}) = {{{}, {}}};\n'.format(id=id, *edge.verts))

            for id, face in zip(self.faces.keys(), self.faces.values()):
                geo_file.write('Curve Loop ({id}) = {{'.format(id=id*10)+', '.join(map(str, face.edges))+'};\n')
                geo_file.write('Surface ({id}) = {{{id2}}};\n'.format(id=id*10, id2=id*10))

            for line in self.gmsh:
                geo_file.write(line)
            return self.mesh_file_name

    def mesh2D(self, elem_size, mesh_type=None, recombine=True, mesh_file_name=None,
               corner_refine_factor=2., mesh_algo=8, recomb_algo=0):
        self.gmsh=[]
        if mesh_type==None:
            self.gmsh.append('Field[1] = MathEval;\n')
            self.gmsh.append('Field[1].F = "{}";\n'.format(elem_size))
            self.gmsh.append('Background Field = 1;\n')
        elif mesh_type== 'Distance':
            self.gmsh.append('Field[1] = Distance;\n')
            self.gmsh.append('Field[1].NodesList = {{{}}};\n'.format(str(list(self.vertices.keys())).replace(']', '').replace('[','')))
            self.gmsh.append('Field[2] = Threshold;\n')
            self.gmsh.append('Field[2].IField = 1;\n')
            self.gmsh.append('Field[2].LcMin = {};\n'.format(elem_size / corner_refine_factor))
            self.gmsh.append('Field[2].LcMax = {};\n'.format(elem_size))
            self.gmsh.append('Field[2].DistMin = {};\n'.format(elem_size * 4))
            self.gmsh.append('Field[2].DistMax = {};\n'.format(elem_size * 8))
            #self.gmsh.append('Field[3] = Min;\n')
            #self.gmsh.append('Field[3].FieldsList = {2};\n')
            self.gmsh.append('Background Field = 2;\n')
            self.gmsh.append('Mesh.CharacteristicLengthExtendFromBoundary = 0;\n')
            self.gmsh.append('Mesh.CharacteristicLengthFromPoints = 0;\n')
            self.gmsh.append('Mesh.CharacteristicLengthFromCurvature = 0;\n')
        self.gmsh.append('Mesh.Algorithm = {};\n'.format(mesh_algo)) #6
        self.gmsh.append('Mesh.Smoothing = 3;\n')
        if recombine==True:
            self.gmsh.append('Mesh.RecombinationAlgorithm = {};\n'.format(recomb_algo))
            self.gmsh.append('Recombine Surface {:};\n')
            self.gmsh.append('Recombine Surface {:};\n')
            self.gmsh.append('Mesh.Smoothing = 3;\n')
        self.write_geo(mesh_file_name)
        subprocess.run('gmsh {}'.format(self.mesh_file_name.rsplit('.',1)[0])+'.geo -2 -nt 2 -format key')
        return self.mesh_file_name.rsplit('.')[0]+'.key'

    def get_statistics(self):
        file_base_name=self.tess_file_name.rsplit('\\')[-1].split('.')[0]
        if len(self.tess_file_name.split('\\'))==1:
            file_list=os.listdir()
            directory=''
        else:
            directory = self.tess_file_name.rsplit('\\',1)[0]+'\\'
            file_list=os.listdir(self.tess_file_name.rsplit('\\',1)[0])

        for file_name in [file_name for file_name in file_list if file_base_name in file_name]:
            if '.stedge' in file_name:
                with open(directory+file_name) as file:
                    self.statEdges = np.array([float(item) for item in file.readlines()])

            if '.stface' in file_name:
                with open(directory+file_name) as file:
                    temp = np.array([[float(value) for value in item.split()] for item in file.readlines()])
                self.statFacearea = temp[:, 0]
                self.statFaceednum = temp[:, 1]

            if '.stpoly' in file_name:
                with open(directory+file_name) as file:
                    temp = np.array([[float(value) for value in item.split()] for item in file.readlines()])

                self.stat_polyvol = temp[:, 0]
                self.stat_polyspher = temp[:, 1]
                self.stat_polyfacenum = temp[:, 2]

    def plot_statistics(self):
        fig, axarr = plt.subplots(2, 2)
        axarr[0, 0].hist(self.statEdges, bins=20)
        axarr[0, 0].scatter([np.average(self.statEdges)], [0], color='red')
        axarr[0, 0].set_title('Edge length')
        axarr[0, 0].set_xlabel('mm')
        axarr[0, 1].hist(self.statFacearea, bins=20)
        axarr[0, 1].scatter([np.average(self.statFacearea)], [0], color='red')
        axarr[0, 1].set_title('Face area')
        axarr[0, 1].set_xlabel('mm$^2$')
        axarr[1, 0].hist(self.stat_polyvol, bins=20)
        axarr[1, 0].scatter([np.average(self.stat_polyvol)], [0], color='red')
        axarr[1, 0].set_title('Cell volume')
        axarr[1, 0].set_xlabel('mm$^3$')
        axarr[1, 1].hist(self.stat_polyspher, bins=20)
        axarr[1, 1].scatter([np.average(self.stat_polyspher)], [0], color='red')
        axarr[1, 1].set_title('Sphericity')
        axarr[1, 1].set_xlabel('[]')
        fig.tight_layout()


#folderName = r'H:\thesis\periodic\representative\S05R20\ID1'
#mesh_file_name = folderName + r'\\test'
#self = Tessellation(folderName + r'\\nfrom_morpho-id1.tess')
#self.regularize(n=int(len(self.edges.keys())/2))
#self.mesh_file_name=mesh_file_name
#self.mesh2D(elem_size=0.02)


#folderName = r'H:\thesis\linear\representative\S05R1\ID1'
#mesh_file_name = folderName + r'\\test'
#self = Tessellation(folderName + r'\\nfrom_morpho-id1.tess')
#self.mesh_file_name=mesh_file_name
#self.mesh2D(elem_size=0.02)
