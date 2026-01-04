import networkx as nx
import matplotlib.pyplot as plt
import random, math
from easydict import EasyDict as edict
import yaml
# import Node

config = edict(yaml.safe_load(open('config.yaml', 'r')))
print("This is FWDGraph")



def drawGraph(G, nodeSet):
    # plt.rcParams['figure.figsize'] = (500, 500)
    # print(G.nodes)
    # print(G.edges)
    # print(G.adj)
    # print(G.degree)
    # nx.draw(G, with_labels=True)
    pos = {}
    pos['s'] = [0,config.map[1]/2]
    pos['t'] = [config.map[0],config.map[1]/2]
    for i in range (0,config.nodeNumber):
        pos[i+1] = nodeSet[i].location
    print(pos)
    # pos = nx.random_layout(G)
    # weights = G.edges.data("weight")
    # print("!!!!!!!!!!!!!!!!")
    # for k in weights:
        # G.add_edge(k[0], k[1], weight=k[2])
    # weights = nx.get_edge_attributes(G, "weight")
    nx.draw(G, pos)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()


def constructGraph(nodeSet):
    # sourceNode = Node.Node([0,config.map[1]/2], config.sensingRange, config.transmissionRange, config.initialEnergy)
    # terminalNode = Node.Node([config.map[0],config.map[1]/2], config.sensingRange, config.transmissionRange, config.initialEnergy)

    G = nx.Graph()
    G.add_nodes_from(['s', 't'])
    alphaV = []

    # 该循环是将冗余节点找出来
    for i in range (config.nodeNumber):
        for j in range(config.nodeNumber):
            if i != j & nodeSet[i].isRedundancy == False & nodeSet[j].isRedundancy == False:
                if isCoincide(nodeSet, i, j) == True:
                    nodeSet[i].redundancySet.append(j)
                    nodeSet[j].isRedundancy = True

    # 该循环是构建全权动态图的顶点，冗余节点不参与图的构建
    for i in range (len(nodeSet)):
        if nodeSet[i].isRedundancy == True:              # 如果这个点是冗余节点，则将alpha值赋值-1
            alphaV.append(-1)
        if nodeSet[i].isRedundancy == False:             # 如果这个点不是冗余节点，则将alpha值赋值其冗余值，也可能是0
            G.add_node(nodeSet[i].ID)
            alphaV.append(len(nodeSet[i].redundancySet))
            if nodeSet[i].location[0] < config.sensingRange:
                G.add_edge('s', i+1, weight=1)
            if config.map[0] - nodeSet[i].location[0] < config.sensingRange:
                G.add_edge(i+1, 't', weight=1)


    # 该循环是构建全权动态图的边，冗余节点不参与图的构建
    for i in range(config.nodeNumber):
        if nodeSet[i].isRedundancy == False:   # 如果该节点是冗余节点，则不参与节点构建
            for j in range(config.nodeNumber):
                if i != j & nodeSet[j].isRedundancy == False:
                    mu = calculateMu(nodeSet, i, j, 0)
                    mu = round(mu, 2)   # 保留两位小数
                    if mu != 0:
                        G.add_edge(i+1, j+1, weight=mu)

    return G,alphaV


def isCoincide(nodeSet, index1, index2):
    d = nodeSet[index1].distanceWithNode(nodeSet[index2])
    r = config.sensingRange
    if d < r:
        # print("Coincide")
        S = r/math.pi*math.acos(pow(d,2)/(2*pow(r,2))-1)-2*d*pow((pow(r,2)-pow(d,2)/4),0.5)
    else:
        # print("Not coincide")
        S = 0
    SCircle = math.pi*config.sensingRange*config.sensingRange
    if 0.9*SCircle <= S:
        return True
    else:
        return False

def calculateAlpha(nodeSet, index, T):
    beta1 = calculateBeta1(nodeSet, index, T)
    chi = len(nodeSet[index].redundancySet)
    if chi == 0:
        return 0
    else:
        return beta1 * chi


def calculateMu(nodeSet, index1, index2, T):
    beta2, delta = calculateBeta2(nodeSet, index1, index2, T)
    if delta == 0:
        return 0
    elif delta == 2 * config.sensingRange:
        return 1
    else:
        return beta2 * (delta/config.sensingRange/2)


def calculateBeta1(nodeSet, index, T):
    if nodeSet[index].resEnergy(T) > config.Et:
        return 1
    else:
        return 1 - (1/len(nodeSet[index].redundancySet))


def calculateBeta2(nodeSet, index1, index2, T):
    # calculate delta
    d = nodeSet[index1].distanceWithNode(nodeSet[index2])
    if d == 0:
        delta = 2 * config.sensingRange
    elif d < 2 * config.sensingRange:
        delta = 2 * config.sensingRange - d
    else:
        delta = 0

    # calculate beta2
    n = T                               # time slot, associated with T
    if T == 0:
        beta2 = 1
        return beta2, delta
    ESd = random.random()
    estimateP_e = 1 - pow((1 - config.P_e), n)
    ESt = estimateP_e * config.Es * T
    beta2 = 1
    if delta!= 0:
        if ESd > ESt:
            beta2 = (delta - 1) / delta
        else:
            beta2 = 1
    return beta2, delta


def FWDGraph(nodeSet, T):
    G, alphaV = constructGraph(nodeSet)
    drawGraph(G, nodeSet)
    return G, alphaV

