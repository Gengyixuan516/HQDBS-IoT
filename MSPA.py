# 本算法是MSPA算法，本质上和Dijkstra算法区别不大，就是将所有节点建模为一个图，然后不断选择当前最短路径构建，被选择过的节点不可以再进行选择，这保证了栅栏是节点不相交路径
# Achieving Crossed Strong Barrier Coverage in Wireless Sensor Network---------Algorithm 1

import networkModel
import numpy as np
import math
import networkx as nx
from easydict import EasyDict as edict
import yaml


with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = edict(yaml.safe_load(file))

np.set_printoptions(threshold=np.inf)


def generateGraph(nodeSet):
    # 's'是0号，'t'是config.nodeNumber+1号
    adjlist = np.zeros((config.nodeNumber+2,config.nodeNumber+2)) # 邻接矩阵

    for i in range (1, config.nodeNumber+1):
        if nodeSet[i-1].location[0] < config.sensingRange:
            adjlist[0,i] = 1
        if config.map[0] - nodeSet[i-1].location[0] < config.sensingRange:
            adjlist[i,config.nodeNumber+1] = 1
        for j in range (1, config.nodeNumber+1):
            if haveEdge(i, j, nodeSet):
                adjlist[i,j] = 1

    # 输出邻接矩阵
    # print('The adjacent matrix is: ')
    # print(adjlist)


    G = nx.Graph()  # 无向图
    # 把点加到图中
    G.add_node('s')
    for i in range(1, config.nodeNumber+1):
        G.add_node(i)
    G.add_node('t')
    # 将边构建起来
    for i in range (1, config.nodeNumber+1):
        if nodeSet[i-1].location[0] < config.sensingRange:
            G.add_edge('s', i)
        if config.map[0] - nodeSet[i-1].location[0] < config.sensingRange:
            G.add_edge(i, 't')
        for j in range (1, config.nodeNumber+1):
            if haveEdge(i, j, nodeSet) & (i != j):
                G.add_edge(i, j)

    return G, adjlist


# 判断点i与点j是之间是否存在边
def haveEdge(i, j, nodeSet):
    d2 = pow((nodeSet[i-1].location[0]-nodeSet[j-1].location[0]),2) + pow((nodeSet[i-1].location[1]-nodeSet[j-1].location[1]),2)
    distance = math.sqrt(d2)
    if distance < config.sensingRange * 2:
        return True
    return False



# Input: The coverage graph G of a WSN
# Output: The maximum number of the barrier paths and the nodes of each barrier path

# if __name__ == '__main__':
#     nodeSet, clusterHeadSet, clusterSet = networkModel.networkModel()



def MSPA(nodeSet):
    G, adjlist = generateGraph(nodeSet)
    pathNodeList = []
    pathCount = 0
    pathFlag = 0

    while 1:
        try:
            nodeList = nx.dijkstra_path(G, 's', 't')
            print('The activated path is :', nodeList)
            pathNodeList.append(nodeList)
            nodeInThisTurn = nodeList[1:len(nodeList)-1]
            G.remove_nodes_from(nodeInThisTurn)
            # print(G.nodes)
            pathCount = pathCount + 1
        except:
            print('Can not find a path any more...')
            break

    print('Now the barrier path comes to: ', end='')
    print(pathCount)
    return pathCount, pathNodeList




def MSPAWithT(nodeSet, G, intrudedNodes):
    # print("Time: ", t)
    pathNodeList = []
    pathCount = 0
    pathFlag = 0
    # 检查节点状态，将不符合要求的节点剔除
    # print(type(G.nodes))
    for node in intrudedNodes:
        if node in G.nodes.keys():
            G.remove_node(node)
    for node in list(G.nodes.keys()):
        if node != 's' and node != 't':
            if nodeSet[node-1].Energy < config.energyThreshold:
                nodeSet[node-1].status = 4
            if nodeSet[node-1].status == 4:
                G.remove_node(node)
            # 将所有满足条件的节点设置为standby状态，供后续进行状态分配
            if nodeSet[node-1].status == 1 or nodeSet[node-1].status == 2:
                nodeSet[node-1].status = 3

    try:
        # step:1 Create the graph
        nodeList = nx.dijkstra_path(G, 's', 't')
        print('The activated path is :', nodeList)
        pathNodeList.append(nodeList)
        nodeInThisTurn = nodeList[1:len(nodeList)-1]

        # step2: Malicious node detection
            # Can the node justify if it is intruded by itself or not?
            # 此处可以通过函数参数传入本次损坏的节点ID，然后进行状态标记，如果没有损坏节点，传入空数组

        # step3: Status update(energy update)
        # update 1 status
        for node in nodeInThisTurn:
            nodeSet[node-1].status = 1
        # update 2 status

        # update energy
        for node in list(G.nodes.keys()):
            if node != 's' and node != 't':
                nodeSet[node-1].updateEnergy()
                if nodeSet[node-1].status == 0:
                    nodeSet[node-1].status = 3

        print('The selected path in this turn is: ')
        print(nodeInThisTurn)

        pathFlag = 1
        return pathFlag, nodeInThisTurn
    except:
        print('Can not find a path any more...')
        pathFlag = 0
        return pathFlag, []

