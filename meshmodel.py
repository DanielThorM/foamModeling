import tessellations as ts
import copy
import importlib
from collections import namedtuple
import numpy as np
import math
#importlib.reload(ts)
#folderName = r'H:\thesis\periodic\representative\S05R1\ID1'
#mesh_file_name = folderName + r'\\test'
#tessellation  = ts.PeriodicTessellation(folderName + r'\\nfrom_morpho-id1.tess')
#tessellation .regularize(n=int(len(self.edges.keys())/2))
#tessellation .mesh_file_name=mesh_file_name
#tessellation .mesh2D(elem_size=0.02)


class PeriodicLSDynaGeometry(object):
    '''Takes inn a tessGeom object that must have been meshed and have an assigned meshFileName'''
    def __init__(self, tessellation, debug=False):
        self.tessellation=copy.deepcopy(tessellation)
        self.nodes, self.shell_elements = self.load_mesh()
        if debug==False:
            self.parts = self.find_parts()
            self.vertex_to_node, self.node_to_vertex = self.find_vertex_node_map()
            self.nodes_on_edges = self.find_nodes_on_edges()
            self.vertex_nodes_master_to, self.vertex_nodes_slave_to = self.find_vertex_nodes_periodicity()
            self.edge_nodes_master_to, self.edge_nodes_slave_to = self.find_edge_nodes_periodicity()
            self.face_node_master_to, self.face_node_slave_to = self.transfer_surfaces()
            self.domain_size = self.tessellation.domain_size

            self.beam_elements = self.find_beam_elements()
            self.beam_parts=self.find_beam_parts()

            self.surface_thickness={}
            self.beam_areas={}

            self.shell_areas = self.find_shell_areas()
            self.beam_lengths = self.beamPartLength()

            self.slave_surfaces = self.find_slave_surfaces()
            self.slave_beams = self.find_slave_beams()
            self.beam_oriention = self.find_beam_oriention()
            self.reference_elements, self.reference_nodes = self.find_reference_elements()

    def load_mesh(self):
        self.node_tup = namedtuple('node', ['n', 'coords'])
        self.element_tup = namedtuple('element', ['elem', 'part', 'nodes'])
        self.part_tup = namedtuple('part', ['pid', 'elems'])

        with open(self.tessellation.mesh_file_name.rsplit('.')[0] + '.key', 'r') as read_file:
            lines = read_file.readlines()
        keyword_ind = [i for i, line in enumerate(lines) if '*' in line]

        ################################################################################
        # Read Nodes
        ################################################################################
        node_keyword_ind = np.where(np.array(lines, dtype=object) == '*NODE\n')[0][0]
        node_last_index = keyword_ind[keyword_ind.index(node_keyword_ind) + 1]
        node_dict = {}
        for line in lines[node_keyword_ind + 1:node_last_index]:
            if '$' not in line:
                if ',' in line:
                    node_dict[int(line.replace('\n', '').split(',')[0])] = self.node_tup(
                        int(line.replace('\n', '').split(',')[0]),  np.array([float(item) for item in line.replace('\n', '').split(',')[1:]]))
                elif isinstance(int(line.split()[0]), int):
                    node_dict[int(line.replace('\n', '').split()[0])] =self.node_tup(
                        int(line.replace('\n', '').split()[0]),  np.array([float(item) for item in line.replace('\n', '').split()[1:]]))
                else:
                    raise Exception('Unexpected value in list of nodes')
        ################################################################################
        # Read Elements
        ################################################################################
        elem_keyword_ind = np.where(np.array(lines, dtype=object) == '*ELEMENT_SHELL\n')[0]
        elem_last_ind = keyword_ind[keyword_ind.index(elem_keyword_ind[-1]) + 1]
        elem_dict = {}
        for line in lines[elem_keyword_ind[0] + 1:elem_last_ind]:
            if '$' not in line and '*' not in line:
                if ',' in line:
                    elem_dict[int(line.replace('\n', '').split(',')[0])] = self.element_tup(
                        *[int(item) for item in line.replace('\n', '').split(',')[:2]],
                        [int(item) for item in line.replace('\n', '').split(',')[2:]])
                elif isinstance(int(line.split()[0]), int):
                    elem_dict[int(line.replace('\n', '').split()[0])] = self.element_tup(
                        *[int(item) for item in line.replace('\n', '').split()[:2]],
                        [int(item) for item in line.replace('\n', '').split()[2:]])
                else:
                    raise Exception('Unexpected value in list of elements')

        return node_dict, elem_dict

    def find_parts(self):
        part_list = set([element.part for element in self.shell_elements.values()])
        part_dict = {}
        for part in part_list:
            part_dict[part] = self.part_tup(part, [element for element in self.shell_elements.values() if element.part == part])
        return part_dict

    def compare_arrays(self, arr0, arr1, rel_tol=1e-07, abs_tol=0.0):
        return all([math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(arr0, arr1)])

    def find_vertex_node_map(self):
        vertex_to_node={}
        for i, vertex in enumerate(self.tessellation.vertices.values()):
            vertex_to_node[vertex.ver_id] = i+1
            if all([math.isclose(a, b, rel_tol=1e-09, abs_tol=1e-9) for a, b in zip(vertex.coords, self.nodes[i + 1].coords)]) == False:
                raise Exception('Vertex {} and node {} location not equal'.format(vertex.ver_id, i+1))
        nodeToVertex = dict(zip(vertex_to_node.values(), vertex_to_node.keys()))
        if len(vertex_to_node) != len(self.tessellation.vertices.keys()):
            raise Exception('All vertexes not mapped')
        return vertex_to_node, nodeToVertex

    def find_vertex_nodes_periodicity(self):
        vertexNodeMasterTo = {}
        vertexNodeSlaveTo = {}
        for vertex in self.tessellation.vertices.values(): #vertex = list(self.tessellation.vertices.values())[0]
            if vertex.masterTo != []:
                vertexNodeMasterTo[self.vertex_to_node[vertex.ver_id]] = []
                for i in range(0, len(vertex.masterTo), 4):
                    vertexNodeMasterTo[self.vertex_to_node[vertex.ver_id]].append([self.vertex_to_node[vertex.masterTo[i]]] + vertex.masterTo[i + 1: i + 4])
                    vertexNodeSlaveTo[self.vertex_to_node[vertex.masterTo[i]]] = [self.vertex_to_node[vertex.ver_id]] + vertex.masterTo[i + 1: i + 4]
        return vertexNodeMasterTo, vertexNodeSlaveTo

    def find_nodes_on_edges(self):
        edgeDict = {}
        #edgeElemCounter = max(self.shell_elements.keys()) + 1
        for edge in self.tessellation.edges.values(): #edge = self.tessellation.edges[1] # Fix when two edges share the same surface
            surfNodes = []
            for surf in edge.parents: #surf=edge.parents[0]
                tempList = []
                for elem in self.parts[surf * 10 + 2000000].elems:
                    tempList.extend(elem.nodes)
                surfNodes.append(tempList)
            nodesOnEdge = list(set(surfNodes[0]).intersection(set(surfNodes[1])))
            for node in nodesOnEdge:
                if all([math.isclose(a, b, rel_tol=1e-08, abs_tol=1e-8) for a, b in
                        zip(self.nodes[node].coords, edge.x0())]) == True:
                    origoNode = node
            filteredNodesOnEdge=[origoNode]
            edgeVector=edge.vector()/np.linalg.norm(edge.vector())
            nodesOnEdge.remove(origoNode)
            for node in nodesOnEdge:
                nodePairVector = self.nodes[node].coords - self.nodes[origoNode].coords
                normNodePairVector = nodePairVector/np.linalg.norm(nodePairVector)
                if all([math.isclose(a, b, rel_tol=1e-08, abs_tol=1e-8) for a, b in
                        zip(abs(edgeVector), abs(normNodePairVector))]) == True:
                    filteredNodesOnEdge.append(node)
            edgeDict[edge.edge_id] = filteredNodesOnEdge
        return edgeDict  #self.tessellation.plotFaces([face for edgeId in self.tessellation.vertices[2114].parents+self.tessellation.vertices[1634].parents for face in self.tessellation.edges[edgeId].parents])

    def find_edge_nodes_periodicity(self):
        nodeMasterTo={}
        nodeSlaveTo={}
        for edge in self.tessellation.edges.values(): #edge = self.tessellation.edges[10]
            if edge.masterTo != []:
                masterNodes = self.nodes_on_edges[edge.edge_id]
                for masterNode in masterNodes:
                    nodeMasterTo[masterNode] = []
                for i in range(0, len(edge.masterTo), 5):
                    periodicity = np.array(edge.masterTo[i+1:i+4])*self.tessellation.domain
                    slaveNodes = self.nodes_on_edges[edge.masterTo[i]]
                    for slaveNode in slaveNodes:
                        nodeSlaveTo[slaveNode] = []
                    counter = 0
                    for masterNode in masterNodes: #masterNode = masterNodes[0]
                        for slaveNode in slaveNodes:  #slaveNode =  slaveNodes[5]
                            #print(self.nodes[slaveNode].coords + periodicity)
                            if all([math.isclose(a, b, rel_tol=1e-08, abs_tol=1e-8) for a, b in
                                    zip(self.nodes[masterNode].coords, self.nodes[slaveNode].coords - periodicity)]) == True:
                                nodeMasterTo[masterNode].append([slaveNode]+edge.masterTo[i+1:i+4])
                                if nodeSlaveTo[slaveNode] != []:
                                    raise Exception('Slave node {} referenced twice'.format(slaveNode))
                                nodeSlaveTo[slaveNode] = [masterNode]+edge.masterTo[i+1:i+4]
                                counter +=1
                    if counter != len(masterNodes):
                        raise Exception('Unequal number of nodes in master edge {} and slave edge {}'.format(edge.edge_id,edge.masterTo[i]))
        return nodeMasterTo, nodeSlaveTo

    def transfer_surfaces(self):
        faceNodeMasterTo = {}
        faceNodeSlaveTo = {}
        for face in self.tessellation.faces.values(): #face = list(self.tessellation.faces.values())[0] #face =self.tessellation.faces[54]
            if face.masterTo != []:
                periodicity = np.array(face.masterTo[1:4])*self.tessellation.domain
                masterPart = self.parts[face.face_id * 10 + 2000000]
                masterPartNodesOnEdge = list(set(
                    [node for edge in self.tessellation.faces[face.face_id].edges for node in
                     self.nodes_on_edges[abs(edge)]]))
                slavePartID = face.masterTo[0] * 10 + 2000000
                slavePart = self.parts[slavePartID]
                slavePartNodesOnEdge = list(set([node for edge in self.tessellation.faces[face.masterTo[0]].edges for node in self.nodes_on_edges[abs(edge)]]))
                lastElementKey = max(self.shell_elements.keys())
                lastNodeKey = max(self.nodes.keys())
                allNodesOnMasterFace=list(set([node for elem in masterPart.elems for node in elem.nodes]))
                allNodesOnSlaveFace = list(set([node for elem in slavePart.elems for node in elem.nodes]))
                newNodeMapDict = {}
                for masterNode in allNodesOnMasterFace:
                    lastNodeKey += 1
                    newNodeMapDict[masterNode] = lastNodeKey
                for masterNode in masterPartNodesOnEdge:
                    for slaveNode in slavePartNodesOnEdge:
                        if all([math.isclose(a, b, rel_tol=1e-09, abs_tol=1e-9) for a, b in
                                zip(self.nodes[masterNode].coords,
                                    self.nodes[slaveNode].coords - periodicity)]) == True:
                            newNodeMapDict[masterNode] = slaveNode


                for masterNode in newNodeMapDict.keys():
                    if masterNode not in masterPartNodesOnEdge:
                            faceNodeMasterTo[masterNode] = [newNodeMapDict[masterNode]] + list(map(int, np.sign(periodicity)))
                            faceNodeSlaveTo[newNodeMapDict[masterNode]] = [masterNode] + list(map(int, np.sign(periodicity)))

                for melement in masterPart.elems: # melement = masterPart.elems[0]
                    lastElementKey +=1
                    newNodeList = []
                    for i, mnode in enumerate(melement.nodes): # mnode = melement.nodes[0]
                        if newNodeMapDict[mnode] not in self.nodes.keys():
                            self.nodes[newNodeMapDict[mnode]] = self.node_tup(newNodeMapDict[mnode],
                                                                              self.nodes[mnode].coords + periodicity)
                        newNodeList.append(newNodeMapDict[mnode])

                    def oldFunct():
                        for i, mnode in enumerate(melement.nodes): #mnode = melement.nodes[2]
                            if mnode in masterPartNodesOnEdge:
                                if mnode in self.edge_nodes_master_to.keys():
                                    periodic=False
                                    for snode in self.edge_nodes_master_to[mnode]:# snode = self.edge_nodes_master_to[mnode][0]
                                        if snode[1:4] == list(periodicity):
                                            newNodeList.append(snode[0])
                                            periodic = True
                                    if periodic == False:
                                        lastNodeKey += 1
                                        newNodeList.append(lastNodeKey)
                                        self.nodes[lastNodeKey] = self.node_tup(lastNodeKey,
                                                                                self.nodes[mnode].coords + periodicity)
                                        faceNodeMasterTo[mnode] = [lastNodeKey] + list(map(int, np.sign(periodicity)))
                                        faceNodeSlaveTo[lastNodeKey] = [mnode] + list(map(int, np.sign(periodicity)))
                                elif mnode in self.edge_nodes_slave_to.keys():
                                    periodic = False
                                    for snode in [self.edge_nodes_slave_to[mnode]]:
                                        if snode[1:4] == list(periodicity):
                                            newNodeList.append(snode[0])
                                            periodic = True
                                    if periodic == False:
                                        lastNodeKey += 1
                                        newNodeList.append(lastNodeKey)
                                        self.nodes[lastNodeKey] = self.node_tup(lastNodeKey,
                                                                                self.nodes[
                                                                                      mnode].coords + periodicity)
                                        faceNodeMasterTo[mnode] = [lastNodeKey] + list(periodicity)
                                        faceNodeSlaveTo[lastNodeKey] = [mnode] + list(periodicity)
                                else:
                                    raise Exception('Node on edge not part of edge periodicity')

                            else:
                                lastNodeKey+=1
                                newNodeList.append(lastNodeKey)
                                self.nodes[lastNodeKey] = self.node_tup(lastNodeKey, self.nodes[mnode].coords + periodicity)
                                faceNodeMasterTo[mnode] = [lastNodeKey] + list(periodicity)
                                faceNodeSlaveTo[lastNodeKey] = [mnode] + list(periodicity)
                            if mnode in melement.nodes[:i]:
                                newNodeList[i] = newNodeList[i-1]
                        if len(newNodeList) != 4:
                            raise Exception('NewNodeList is too short! Investigate')

                    if len(newNodeList) != 4:
                        raise Exception('NewNodeList is too short! Investigate')
                    self.shell_elements[lastElementKey] = self.element_tup(lastElementKey, slavePartID, newNodeList) #namedtuple('element', ['elem', 'part', 'nodes'])

                for element in slavePart.elems:
                    del self.shell_elements[element.elem]
                for node in allNodesOnSlaveFace:
                    if node not in slavePartNodesOnEdge:
                        del self.nodes[node]
                self.parts[slavePartID] = self.part_tup(slavePartID, [element for element in self.shell_elements.values() if element.part == slavePartID])

                if set(slavePartNodesOnEdge).intersection(set([node for elem in self.parts[slavePartID].elems for node in elem.nodes])) != set(slavePartNodesOnEdge):
                    raise Exception('SlaveNodes not preserved')


        #Remove vertexnodes from edgeRelations
        for masterNode, masterTo in zip(self.vertex_nodes_master_to.keys(), self.vertex_nodes_master_to.values()): #masterNode, masterTo =  list(zip(self.vertex_nodes_master_to.keys(), self.vertex_nodes_master_to.values()))[0]
            if masterNode in self.edge_nodes_master_to.keys():
                del self.edge_nodes_master_to[masterNode]
            if masterNode in self.edge_nodes_slave_to.keys():
                del self.edge_nodes_slave_to[masterNode]
            for slaveNode in masterTo: #slaveNode = masterTo[0]
                if slaveNode[0] in self.edge_nodes_slave_to.keys():
                    del self.edge_nodes_slave_to[slaveNode[0]]
                if slaveNode[0] in self.edge_nodes_master_to.keys():
                    del self.edge_nodes_master_to[slaveNode[0]]

        return faceNodeMasterTo, faceNodeSlaveTo

    def checkNodePeriodicity(self):
        masterNodeList, slaveNodeList = [], []
        for node, slaveNodes in zip(self.vertex_nodes_master_to.keys(), self.vertex_nodes_master_to.values()):  # vertex = list(self.vertices.values())[0]
            masterNodeList.append(node)
            slaveNodeList.extend([slaveNode[0] for slaveNode in slaveNodes])
        if len(set(list(self.vertex_nodes_slave_to.keys())).intersection(set(slaveNodeList))) != len(set(slaveNodeList)):
            raise Exception('Vertex node dependencies not resolved')

        masterNodeList, slaveNodeList = [], []
        for node, slaveNodes in zip(self.edge_nodes_master_to.keys(),
                                    self.edge_nodes_master_to.values()):  # vertex = list(self.vertices.values())[0]
            masterNodeList.append(node)
            slaveNodeList.extend([slaveNode[0] for slaveNode in slaveNodes])
        if len(set(list(self.edge_nodes_slave_to.keys())).intersection(set(slaveNodeList))) != len(set(slaveNodeList)):
            raise Exception('Edge dependencies not resolved')

    def find_beam_elements(self):
        beamElementDict = {}
        edgeElemCounter = max(self.shell_elements.keys()) + 1
        for edge in self.tessellation.edges.values(): #edge = list(self.tessellation.edges.values())[0]
            exampleSurfID = edge.parents[0]
            nodesOnEdge = self.nodes_on_edges[edge.edge_id]
            #elemList = []
            for elem in self.parts[exampleSurfID * 10 + 2000000].elems: # elem =  self.parts[exampleSurfID  * 10 + 2000000].elems[0]
                intersectingNodes = set(nodesOnEdge).intersection(set(elem.nodes))
                if len(intersectingNodes) >= 2:
                    #elemList.append(self.element_tup(edgeElemCounter, edge.edge_id + 4000000, list(intersectingNodes)))
                    beamElementDict[edgeElemCounter] = self.element_tup(edgeElemCounter, edge.edge_id + 4000000,
                                                                        intersectingNodes)
                    edgeElemCounter += 1
        return beamElementDict

    def find_beam_parts(self):
        partList = set([element.part for element in self.beam_elements.values()])
        partDict = {}
        for part in partList:
            partDict[part] = self.part_tup(part, [element for element in self.beam_elements.values() if
                                                  element.part == part])
        return partDict

    def find_beam_oriention(self):
        orientDict={}
        for beamPart in self.beam_parts.keys():
            #Find connected edge in random vertex:
            edge_id = beamPart-4000000
            edge_center = np.mean([self.tessellation.vertices[ver_id].coords for ver_id in self.tessellation.edges[edge_id].edge_vers], axis=0)
            parentSurfaces= self.tessellation.edges[edge_id].parents
            edgeVector = self.tessellation.faces[parentSurfaces[0]].find_barycenter() - edge_center
            orientDict[beamPart] = edgeVector
        return orientDict
    ##########################################################################
    #Set part qualities
    ##########################################################################

    def setSurfThickness(self, tt, pid=None):
        if pid==None:
            for pid in self.parts.keys():
                self.parts[pid]['thickness'] = tt
        else:
            self.parts[pid]['thickness'] = tt

    def setBeamArea(self, aa, pid=None):
        if pid==None:
            for pid in self.beamDict.keys():
                self.beamDict[pid]['csArea'] = aa
        else:
            self.beamDict[pid]['csArea'] = aa

    ##########################################################################
    # Calculate qualities
    ##########################################################################
    def find_slave_surfaces(self):
        return [slavePartID * 10 + 2000000 for face in self.tessellation.faces.values() for slavePartID in face.masterTo[::5]]
    def find_slave_beams(self):
        return [slavePartID * 10 + 2000000 for face in self.tessellation.faces.values() for slavePartID in face.masterTo[::5]]
    def relativeDensity(self):
        solidVolume=sum([self.shell_areas[part.pid] * self.surface_thickness[part.pid] for part in self.parts.values() if part.pid not in self.slave_surfaces])
        solidVolume+=sum([self.beam_lengths[part.pid] * self.beam_areas[part.pid] for part in self.beam_parts.values() if part.pid not in self.slave_beams]) # if part.pid not in set(edgeBeamPartsX).intersection(edgeBeamPartsY) and part.pid not in edgeBeamPartsZ]) #Fix this
        totalVolume = np.prod(self.domain_size)
        return solidVolume/totalVolume

    def surfFraction(self):
        surfVolume = sum([self.shell_areas[part.pid] * self.surface_thickness[part.pid] for part in self.parts.values() if part.pid not in self.slave_surfaces])
        solidVolume = surfVolume + sum([self.beam_lengths[part.pid] * self.beam_areas[part.pid] for part in self.beam_parts.values() if part.pid not in self.slave_beams])
        return surfVolume/solidVolume

    def phi(self):
        surfVolume = sum(
            [self.shell_areas[part.pid] * self.surface_thickness[part.pid] for part in self.parts.values() if
             part.pid not in self.slave_surfaces])
        beamVolume= sum([self.beam_lengths[part.pid] * self.beam_areas[part.pid] for part in self.beam_parts.values() if part.pid not in self.slave_beams])
        solidVolume = surfVolume + beamVolume
        return beamVolume/solidVolume

    def find_shell_areas(self):
        areaDict={}
        for part in self.parts.values():
            areaDict[part.pid]=sum([self.shellElementArea(element.elem) for element in part.elems])
        return areaDict

    def shellElementArea(self, element):
        coords=np.array([node.coords for node in [self.nodes[nodeID] for nodeID in self.shell_elements[element].nodes]])
        v1=coords[2]-coords[0]
        v2=coords[3]-coords[1]
        A1=np.linalg.norm(np.cross(v1, v2))/2.

        return abs(A1)

    def beamPartLength(self):
        lenDict={}
        for part in self.beam_parts.values():
            lenDict[part.pid]=sum([self.beamElementLength(element) for element in part.elems])
        return lenDict

    def beamElementLength(self, element):
        coords=np.array([node.coords for node in [self.nodes[nodeID] for nodeID in element.nodes]])
        v1=coords[1]-coords[0]
        L1=np.linalg.norm(v1)
        return L1


    ##########################################################################
    # Find node pairs and directions
    ##########################################################################
    def findNodePairList(self):
        nodePairDict = {1:[], 2:[], 3:[]}
        for face in self.tessellation.faces.values():
            if face.masterTo != []: #self.tessellation.faces[115].masterTo
                if len(face.masterTo) >5:
                    raise Exception('Face {} has more than 1 slave'.format(face.face_id))
                #print(face.masterTo)
                masterFaceID = face.face_id
                slaveFaceID = face.masterTo[0]
                period = np.array(face.masterTo[1:-1])*self.tessellation.domain
                masterNodes =  list(set([node for elems in
                                         self.parts[masterFaceID * 10 + 2000000].elems for node in elems.nodes]))
                masterCoords = [[node, self.nodes[node].coords] for node in masterNodes]
                slaveNodes =  list(set([node for elems in
                                        self.parts[slaveFaceID * 10 + 2000000].elems for node in elems.nodes]))
                slaveCoords = [[node, self.nodes[node].coords - period] for node in slaveNodes]
                matchedNodeList=[]
                for masterNode in masterCoords:
                    minDev = 1
                    slaveNodeID = 0
                    for slaveNode in slaveCoords:
                        if all([math.isclose(a, b, rel_tol=1e-09, abs_tol = 1e-9) for a, b in
                                zip(masterNode[1], slaveNode[1])]) == True:
                            #print ('True for slave {}'.format(slaveNode[0]))
                            if [int(masterNode[0]), int(slaveNode[0])] not in matchedNodeList:
                                matchedNodeList.append([int(masterNode[0]), int(slaveNode[0])])
                        if sum(map(abs,(masterNode[1] - slaveNode[1]))) < minDev:
                            minDev=sum(masterNode[1] - slaveNode[1])
                            slaveNodeID = slaveNode[0]


                #{3164, 3165, 3168, 3602, 3783}
                #set(np.array(masterCoords)[:,0]) - set(np.array(matchedNodeList)[:,0])  10265
                #[14290, 14291, 14294, 14728, 14909]
                #set(np.array(slaveCoords)[:,0]) - set(np.array(matchedNodeList)[:,1])
                # a= [masterCoord for masterCoord in masterCoords if masterCoord[0] in [10265]]
                #b = [slaveCoord for slaveCoord in slaveCoords if slaveCoord[0] in [30099]]

                #9037
                if len(matchedNodeList) != len(masterNodes):
                    raise Exception('Not all nodes equally spaced for face {}'.format(face.face_id))
                for dir, periodSign in zip([1,2,3], period):
                    if dir !=0:
                        if np.sign(periodSign) == -1: #Switch master location if master on wrong side.
                            reversedMatchedNodeList = [a[::-1] for a in matchedNodeList]
                            nodePairDict[dir].extend(reversedMatchedNodeList)
                        else:
                            nodePairDict[dir].extend(matchedNodeList)

    def find_reference_elements(self, refElementSize=0.1):
        nodeCounter = max(self.nodes.keys())
        elemCounter = 1
        partCounter = 6000000 + 10
        refNodeDict = {}
        refElemDict = {}
        elemCounter += 1
        refLoc = np.array([0., 0., 0.])
        nodeList = []
        refElementSize = refElementSize
        elementOffset =  np.array([[0, 0, 0], [refElementSize, 0, 0], [refElementSize, refElementSize, 0], [0, refElementSize, 0], [0, 0, refElementSize], [refElementSize, 0, refElementSize], [refElementSize, refElementSize, refElementSize], [0, refElementSize, refElementSize]])
        for offset in elementOffset:
            nodeCounter += 1
            refNodeDict[nodeCounter] = self.node_tup(nodeCounter, refLoc + offset)
            nodeList.append(nodeCounter)
        refElemDict[elemCounter] = self.element_tup(elemCounter, partCounter, nodeList)
        for i, location in enumerate(self.domain_size): #location = self.domain_size[0]
            elemCounter += 1
            refLoc = np.array([0., 0., 0.])
            refLoc[i] = location
            nodeList = []
            for offset in elementOffset:
                nodeCounter += 1
                refNodeDict[nodeCounter] = self.node_tup(nodeCounter, refLoc + offset)
                nodeList.append(nodeCounter)
            refElemDict[elemCounter] = self.element_tup(elemCounter, partCounter, nodeList)
        return refElemDict, refNodeDict

self=PeriodicLSDynaGeometry(tessellation=tesselation, debug=True)