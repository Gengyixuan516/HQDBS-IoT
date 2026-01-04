import Cluster, Node
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
import yaml

with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = edict(yaml.safe_load(file))



# 生成节点，其中number是生成的节点总数，map是地图尺寸，是个二元组，同时将节点加入到图中
def generateNodes(number, map):
    nodeSet = []
    for i in range(0, number):
        x = random.randint(0, map[0])
        y = random.randint(0, map[1])
        location = [x, y]  # 随机生成节点位置   #！！！！！！！！！！！！！！！！！！！！有待进一步改进！！！！！！！！！！！！！！！！！！！！！！！#
        # print(location)
        nodeSet.append(Node.Node(location, config.sensingRange, config.transmissionRange, config.Ei))  # 将生成的节点加入到节点集里
    nodeSet[0].resetCount()
    return nodeSet


def isInArea(i, j, location):
    left = i * 2 * config.transmissionRange
    right = (i + 1) * 2 * config.transmissionRange
    top = j * 2 * config.transmissionRange
    bottom = (j + 1) * 2 * config.transmissionRange
    flag = True
    if(location[0] <= left or location[0] > right or location[1] <= top or location[1] >= bottom):
        flag = False
    return flag


def selectClusterHead(nodeSet): # 选择虚拟簇头，并返回簇头的集合
    clusterHeadSet = []
    numOfClusterX = (int)(config.map[0] / 2 / config.transmissionRange)
    numOfClusterY = (int)(config.map[1] / 2 / config.transmissionRange)

    # numOfCluster = numOfClusterX * numOfClusterY
    for i in range(0, numOfClusterX):
        for j in range(0, numOfClusterY):
            coordinateX = i * 2 * config.transmissionRange + config.transmissionRange
            coordinateY = j * 2 * config.transmissionRange + config.transmissionRange
            minDis = 2 * config.transmissionRange
            minDisNode = 0
            numOfNode = len(nodeSet)
            for k in range(0, numOfNode):
                distance = math.sqrt(math.pow(coordinateX - nodeSet[k].location[0], 2) + math.pow(
                    coordinateY - nodeSet[k].location[1], 2))
                if(distance <= minDis and isInArea(i, j, nodeSet[k].location) == True): # 找出距离最近的节点
                    minDis = distance
                    minDisNode = nodeSet[k]
            clusterHeadSet.append(minDisNode) # 将该节点加入到簇头中，效果是先列后行
    return clusterHeadSet


def clusterFormation(clusterHeadSet, nodeSet):
    clusterSet = []
    numberOfClusterHead = len(clusterHeadSet)
    for i in range(0, numberOfClusterHead):
        cluster = Cluster.Cluster(clusterHeadSet[i])
        cluster.formation(nodeSet)
        clusterSet.append(cluster)
    return clusterSet


def plot_circle(center, r):
    x = np.linspace(center[0] - r, center[0] + r, 5000)
    y1 = np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
    y2 = -np.sqrt(r ** 2 - (x - center[0]) ** 2) + center[1]
    plt.plot(x, y1, c='k')
    plt.plot(x, y2, c='k')


def networkModel():
    nodeSet = generateNodes(config.nodeNumber, config.map)                    # 生成节点,并返回节点集合
    clusterHeadSet = selectClusterHead(nodeSet)                 # 选择簇头
    clusterSet = clusterFormation(clusterHeadSet, nodeSet)      # 簇的形成
    # return nodeSet, clusterHeadSet, clusterSet

    # ----------------------------plot------------------------------ #
    # plt.figure(dpi=100,figsize=(10,10))
    # plt.xlim(0, config.map[0])
    # plt.ylim(0, config.map[1])
    # xSet = [50,100,150,200,250,300,350,400,450,500]
    # for i in range(0,len(xSet)):
    #     plt.axvline(x=xSet[i], color='k',linewidth=0.5)
    # ySet = [50,100,150,200,250,300,350,400,450,500]
    # for i in range(0,len(ySet)):
    #     plt.axhline(y=ySet[i], color='k',linewidth=0.5)
    # # fig, ax = plt.subplots()
    # for i in range(0, config.nodeNumber):
    #     # circle1 = plt.Circle((nodeSet[i].location[0], nodeSet[i].location[1]), config.sensingRange, color='b', fill=False)
    #     # ax.add_artist(circle1)
    #     plt.text(nodeSet[i].location[0], nodeSet[i].location[1], i+1, ha="center", va="center")
    #     plot_circle((nodeSet[i].location[0], nodeSet[i].location[1]), config.sensingRange)
    #     if nodeSet[i].isSelect == True:
    #         plt.scatter(nodeSet[i].location[0], nodeSet[i].location[1], color='green')
    #     else:
    #         plt.scatter(nodeSet[i].location[0], nodeSet[i].location[1], color='blue')
    # for i in range(0, len(clusterHeadSet)):
    #     if clusterHeadSet[i] != 0:
    #         plt.scatter(clusterHeadSet[i].location[0], clusterHeadSet[i].location[1], color = 'yellow')
    #         # plot_circle((clusterHeadSet[i].location[0], clusterHeadSet[i].location[1]), transmissionRange)
    # plt.savefig('demo.svg')
    # plt.show()

    # -------------------------------------------------------------- #
    return nodeSet, clusterHeadSet, clusterSet



    # for i in range(0, len(clusterSet)):                       # 簇内生成模糊图
    #     print("Cluster: ", i)
    #     fuzzyGraphofCluster.append(fuzzyGraph.create_directed_graph_from_edges(clusterSet[i].nodeSet))
    # path = pathConstruction.constrct(nodeSet)                 # 针对每个簇建立路径
    # for i in range(0, len(clusterSet)):
    #     print("Cluster: ", i)
    #     fuzzyGraph.draw_directed_graph(fuzzyGraphofCluster[i])



