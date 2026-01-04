import Cluster,networkModel

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict
import yaml, math, random


with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = edict(yaml.safe_load(file))


# 图类
# 节点与图之间用ID，唯一标识号传递，在随机生成节点的同时加入到图中
class Graph_Matrix:
    """
    Adjacency Matrix
    """

    def __init__(self, vertices=[], matrix=[]):
        """
        :param vertices:a dict with vertex id and index of matrix , such as {vertex:index}
        :param matrix: a matrix
        """
        self.matrix = matrix
        self.edges_dict = {}  # {(tail, head):weight}
        self.edges_array = []  # (tail, head, weight)
        self.vertices = vertices
        self.num_edges = 0

        # if provide adjacency matrix then create the edges list
        if len(matrix) > 0:
            if len(vertices) != len(matrix):
                raise IndexError
            self.edges = self.getAllEdges()
            self.num_edges = len(self.edges)

        # if do not provide a adjacency matrix, but provide the vertices list, build a matrix with 0
        elif len(vertices) > 0:
            self.matrix = [[0 for col in range(len(vertices))] for row in range(len(vertices))]

        self.num_vertices = len(self.matrix)

    def isOutRange(self, x):
        try:
            if x >= self.num_vertices or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")

    def isEmpty(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices == 0

    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices[key] = len(self.vertices) + 1

        # add a vertex mean add a row and a column
        # add a column for every row
        for i in range(self.getVerticesNumbers()):
            self.matrix[i].append(0)

        self.num_vertices += 1

        nRow = [0] * self.num_vertices
        self.matrix.append(nRow)

    def getVertex(self, key):
        pass

    def add_edges_from_list(self, edges_list):  # edges_list : [(tail, head, weight),()]
        for i in range(len(edges_list)):
            self.add_edge(edges_list[i][0], edges_list[i][1], edges_list[i][2], )

    def add_edge(self, tail, head, cost=0):
        # if self.vertices.index(tail) >= 0:
        #   self.addVertex(tail)
        if tail not in self.vertices:
            self.add_vertex(tail)
        # if self.vertices.index(head) >= 0:
        #   self.addVertex(head)
        if head not in self.vertices:
            self.add_vertex(head)

        # for directory matrix
        self.matrix[self.vertices.index(tail)][self.vertices.index(head)] = cost
        # for non-directory matrix
        # self.matrix[self.vertices.index(fromV)][self.vertices.index(toV)] = \
        #   self.matrix[self.vertices.index(toV)][self.vertices.index(fromV)] = cost

        self.edges_dict[(tail, head)] = cost
        self.edges_array.append((tail, head, cost))
        self.num_edges = len(self.edges_dict)

    def getEdges(self, V):
        pass

    def getVerticesNumbers(self):
        if self.num_vertices == 0:
            self.num_vertices = len(self.matrix)
        return self.num_vertices

    def getAllVertices(self):
        return self.vertices

    def getAllEdges(self):

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                if 0 < self.matrix[i][j] < float('inf'):
                    self.edges_dict[self.vertices[i], self.vertices[j]] = self.matrix[i][j]
                    self.edges_array.append([self.vertices[i], self.vertices[j], self.matrix[i][j]])

        return self.edges_array

    def __repr__(self):
        return str(''.join(str(i) for i in self.matrix))

    def to_do_vertex(self, i):
        print('vertex: %s' % (self.vertices[i]))

    def to_do_edge(self, w, k):
        print('edge tail: %s, edge head: %s, weight: %s' % (self.vertices[w], self.vertices[k], str(self.matrix[w][k])))



# 用边生成有向图
def create_directed_graph_from_edges(nodeSet):
    vertices = []
    #将节点集合的ID加入到vertices中
    for i in range (0, len(nodeSet)):
        vertices.append(nodeSet[i].ID)

    edge_list = []

    myGraph = Graph_Matrix(vertices)
    # 生成边，同时计算其权值
    # 这里将全部阶段遍历，可以建立起每个节点的冗余集和节点之间的边权值
    for i in range (0, len(nodeSet)):
        if nodeSet[i].isVisited == False:
            for j in range (0, len(nodeSet)):
                if i != j and nodeSet[j].isVisited == False:
                    dis = nodeSet[i].distanceWithNode(nodeSet[j])
                    # 定义冗余，当距离小于10%sensing range的时候定义为冗余
                    if(dis < 0.1 * config.sensingRange):
                        nodeSet[i].redundancySet.append(nodeSet[j])
                        nodeSet[j].isVisited = True
                    # 否则，计算传感范围是否overlap，计算边权
                    elif dis <= 2 * config.sensingRange and nodeSet[i].location[0] < nodeSet[j].location[0]:
                        myGraph.add_edge(nodeSet[i].ID, nodeSet[j].ID, calculateMu(nodeSet[i], nodeSet[j]))

    # my_graph.add_edges_from_list(edge_list)
    return myGraph

def generateGraph(nodeSet):
    G = nx.Graph()  # 无向图
    # G.add_node('s')
    for i in range(1, config.nodeNumber + 1):
        G.add_node(i)
    # G.add_node('t')
    for i in range (0, len(nodeSet)):
        if nodeSet[i].isVisited == False:
            # if nodeSet[i].location[0] < config.sensingRange:
            #     G.add_edge('s', nodeSet[i].ID, weight=1)
            # if config.map[0] - nodeSet[i].location[0] < config.sensingRange:
            #     G.add_edge(nodeSet[i].ID, 't', weight=1)
            for j in range (0, len(nodeSet)):
                if i != j and nodeSet[j].isVisited == False:
                    dis = nodeSet[i].distanceWithNode(nodeSet[j])
                    # 定义冗余，当距离小于10%sensing range的时候定义为冗余
                    if(dis < 0.1 * config.sensingRange):
                        nodeSet[i].redundancySet.append(nodeSet[j])
                        nodeSet[j].isVisited = True
                    # 否则，计算传感范围是否overlap，计算边权
                    elif dis <= 2 * config.sensingRange and nodeSet[i].location[0] < nodeSet[j].location[0]:
                        mu = calculateMu(nodeSet[i], nodeSet[j])
                        G.add_edge(nodeSet[i].ID, nodeSet[j].ID, weight=mu)
    return G


# 绘制有向图，带权
def draw_directed_graph(my_graph):
    G = nx.Graph()  # 建立一个空的无向图G

    for node in my_graph.vertices:
        G.add_node(node)

    G.add_weighted_edges_from(my_graph.edges_array)

    print("nodes:", G.nodes())  # 输出全部的节点
    print("edges:", G.edges())  # 输出全部的边
    print("number of edges:", G.number_of_edges())  # 输出边的数量
    print("-------------------------------------------------------")
    nx.draw(G, with_labels=True)
    plt.show()



# 模糊图的F1，点与点之间的关系，即Beta值
def calculateBeta(nodeI):
    tmp = len(nodeI.redundancySet)
    if tmp == 0:
        beta = 0
    else:
        beta = tmp / (tmp + 1)
    return beta


# 模糊图的F2，点与边之间的关系，即Mu值
def calculateMu(nodeI, nodeJ):
    # 计算两节点之间的圆心距
    distance = nodeI.distanceWithNode(nodeJ)
    # 计算delta，以便于后续计算mu
    if distance == 0:
        delta = 2 * config.sensingRange
    elif distance < 2 * config.sensingRange:
        delta = 2 * config.sensingRange - distance
    else:
        delta = 0
    # 计算mu
    if delta == 0:
        mu = 0
    elif delta < 2 * config.sensingRange:
        mu = delta / (2 * config.sensingRange)
    else:
        mu = 1
    # print("mu: ",mu)
    return mu

def select_path_from_fuzzyGraph(myGraph, nodeSet):
    pathSet = []
    node_S = []  # list 每一项存入的是[节点ID，节点权值]
    for node in nodeSet:
        if node.location[0] < config.sensingRange:
            node_S.append([node.ID, calculateBeta(node)])
    while True:
        # print('1')
        path = []
        vi = select_max_beta_node(node_S, nodeSet)
        if vi == 0:
            break
        path.append(vi)
        nodeSet[vi - 1].isVisited = True
        while True:
            # print('2')
            min_E = 1 #这个是指节点之间最小的mu值，选取mu值最小的那个节点，距离最远，收益最好
            vj = 0
            for item in myGraph.edges:
                if item[0] == vi and nodeSet[item[0]-1].location[0]-nodeSet[item[1]-1].location[0]<0 and myGraph.edges[item[0],item[1]]['weight']< min_E and nodeSet[item[1]-1].isVisited == False:
                    vj = item[1]
                    min_E = myGraph.edges[item[0],item[1]]['weight']
                if item[1] == vi and nodeSet[item[1]-1].location[0]-nodeSet[item[0]-1].location[0]<0 and myGraph.edges[item[0],item[1]]['weight']< min_E and nodeSet[item[0]-1].isVisited == False:
                    vj = item[0]
                    min_E = myGraph.edges[item[0],item[1]]['weight']
            if vj == 0: #这种情况就是找不到合适的路了，直接结束
                break
            elif config.map[0] - nodeSet[vj-1].location[0] < config.sensingRange: #这个情况是到达终点，标记所有节点为visited
                path.append(vj)
                # nodeSet[vj - 1].isVisited = True
                for node in path:
                    nodeSet[node-1].isVisited = True
                pathSet.append(path)
                break
            else:
                path.append(vj)
                # nodeSet[vj - 1].isVisited = True
                vi = vj


    return pathSet


# 选择点权最大的节点，返回选择节点的编号
# def select_max_beta_node(node_S, nodeSet):
#     selectedNode = 0
#     max_V = []
#     maxBeta = -1
#     for item in node_S:
#         # print(type(item[1]),type(maxBeta))
#         # print(item[0]-1)
#         if nodeSet[item[0]-1].isVisited == False and item[1] > maxBeta:
#             maxBeta = item[1]
#     for item in node_S:
#         if nodeSet[item[0]-1].isVisited == False and item[1] == maxBeta:
#             max_V.append(item[0])
#     if len(max_V) > 1:
#         selectedNode = random.choice(max_V)
#     return selectedNode
def select_max_beta_node(node_S, nodeSet):
    maxBeta = -1
    max_V = []
    for node_id, beta in node_S:
        if not nodeSet[node_id - 1].isVisited:
            if beta > maxBeta:
                maxBeta = beta
                max_V = [node_id]
            elif beta == maxBeta:
                max_V.append(node_id)

    if not max_V:
        return 0
    return random.choice(max_V)  # 1个也能正常返回


def EC2(nodeSet):
    nodeSet, clusterHeadSet, clusterSet = networkModel.networkModel()
    myGraph = generateGraph(nodeSet)
    pathSet = select_path_from_fuzzyGraph(myGraph, nodeSet)
    for path in pathSet:
        # calculateEfficiencyIndex(path):
        print(path)
    print('Now the EC2 comes to: ', end='')
    print(len(pathSet))
    return len(pathSet), pathSet

def EC2WithT(nodeSet, G, intrudedNodes):

    for node in intrudedNodes:
        # if node in G.nodes.keys():
        if node in G:
            G.remove_node(node)
    for node in list(G.nodes):
        if nodeSet[node - 1].Energy < config.energyThreshold:
            nodeSet[node - 1].status = 4
        if nodeSet[node - 1].status == 4:
            G.remove_node(node)
        # 将所有满足条件的节点设置为standby状态，供后续进行状态分配
        if nodeSet[node - 1].status == 1 and nodeSet[node - 1].status == 2:
            nodeSet[node - 1].status = 3

    pathFlag = 0

    try:
        print("Searching for the EC2...")
        # step1: Find the path
        pathSet = []
        node_S = []  # list 每一项存入的是[节点ID，节点权值]
        for node in nodeSet:
            if node.location[0] < config.sensingRange:
                node_S.append([node.ID, calculateBeta(node)])

        # Select the path
        while len(pathSet) < 1:
            path = []
            vi = select_max_beta_node(node_S, nodeSet)
            if vi == 0:
                return pathFlag, []
            path.append(vi)
            nodeSet[vi - 1].isVisited = True
            while True:
                # print('2')
                min_E = 1  # 这个是指节点之间最小的mu值，选取mu值最小的那个节点，距离最远，收益最好
                vj = 0
                for item in G.edges:
                    if item[0] == vi and nodeSet[item[0] - 1].location[0] - nodeSet[item[1] - 1].location[0] < 0 and \
                            G.edges[item[0], item[1]]['weight'] < min_E and nodeSet[
                        item[1] - 1].isVisited == False:
                        vj = item[1]
                        min_E = G.edges[item[0], item[1]]['weight']
                    if item[1] == vi and nodeSet[item[1] - 1].location[0] - nodeSet[item[0] - 1].location[0] < 0 and \
                            G.edges[item[0], item[1]]['weight'] < min_E and nodeSet[
                        item[0] - 1].isVisited == False:
                        vj = item[0]
                        min_E = G.edges[item[0], item[1]]['weight']
                if vj == 0:  # 这种情况就是找不到合适的路了，直接结束
                    break
                elif config.map[0] - nodeSet[vj - 1].location[0] < config.sensingRange:  # 这个情况是到达终点，标记所有节点为visited
                    path.append(vj)
                    # nodeSet[vj - 1].isVisited = True
                    for node in path:
                        nodeSet[node - 1].isVisited = True
                    pathSet.append(path)
                    pathFlag = 1
                    break
                else:
                    path.append(vj)
                    # nodeSet[vj - 1].isVisited = True
                    vi = vj

        print('The selected path in this turn is: ')
        print(pathSet[0])

        return pathFlag, pathSet[0]


    except Exception as e:
        print(e)
        print("EC2 can not find the path anymore...")
        return pathFlag, []



# pathNum, pathSet = EC2(nodeSet)
# for path in pathSet:
#     print(path)
# print('Finish')
# draw_directed_graph(myGraph)

if __name__ == '__main__':
    nodeSet, clusterHeadSet, clusterSet = networkModel.networkModel()
    myGraph = generateGraph(nodeSet)
    path = select_path_from_fuzzyGraph(myGraph, nodeSet)
    print(path)