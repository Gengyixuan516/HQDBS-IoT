# 本算法是MAX-FLOW算法,设置每条边的capacity = 1

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

    G = nx.DiGraph()  # 有向图
    # 把点加到图中
    G.add_node('s')
    for i in range(1, config.nodeNumber+1):
        G.add_node(i)
    G.add_node('t')
    # 将边构建起来
    for i in range (1, config.nodeNumber+1):
        # 拆点，in为正值，out为负值
        G.add_edge(i, -i, capacity=1)
        if nodeSet[i-1].location[0] < config.sensingRange:
            G.add_edge('s', i, capacity=1)
        if config.map[0] - nodeSet[i-1].location[0] < config.sensingRange:
            G.add_edge(-i, 't', capacity=1)
        for j in range (1, config.nodeNumber+1):
            if haveEdge(i, j, nodeSet) & (i != j):
                G.add_edge(-i, j, capacity=1)
                G.add_edge(-j, i, capacity=1)

    return G


# 判断点i与点j是之间是否存在边
def haveEdge(i, j, nodeSet):
    d2 = pow((nodeSet[i-1].location[0]-nodeSet[j-1].location[0]),2) + pow((nodeSet[i-1].location[1]-nodeSet[j-1].location[1]),2)
    distance = math.sqrt(d2)
    if distance < config.sensingRange * 2:
        return True
    return False


# if __name__ == '__main__':
#     nodeSet, clusterHeadSet, clusterSet = networkModel.networkModel()

def MaxFlow(nodeSet):
    G = generateGraph(nodeSet)
    print('Searching for the max-flow...')
    try:
        flow_value, flow_dict = nx.maximum_flow(G, 's', 't')
    except:
        print('Can not find the max-flow...')

    print('Now the max-flow comes to: ', end='')
    print(flow_value)
    # for key in flow_dict:
    #     print(key, flow_dict[key])
    pathList = []
    for key in flow_dict['s']:
        if flow_dict['s'][key] == 1:
            pathList.append(['s',key])
    for path in pathList:
        while path[-1] != 't':
            currentDict = flow_dict[path[-1]]
            for key in currentDict:
                if currentDict[key] == 1:
                    path.append(key)
                    break
    print('===THE FINAL PATH===')
    for path in pathList:
        # print(path)
        path = path[1:len(path)-1]
        # print(path)
        for element in path:
            if element < 0:
                path.remove(element)
        print(path)

    return flow_value, pathList

def MaxFlowWithT(nodeSet, G, intrudedNodes):
    pathFlag = 0

    # 1) 删除本轮入侵节点
    for node in intrudedNodes:
        if node in G.nodes.keys():
            G.remove_node(node)
    # 2) 能量阈值/失败节点剔除 + sleep 处理
    for node in list(G.nodes.keys()):
        if node != 's' and node != 't' and node > 0:
            if nodeSet[node-1].Energy < config.energyThreshold:
                nodeSet[node-1].status = 4
            if nodeSet[node-1].status == 4:
                G.remove_node(node)
                G.remove_node(-node)
            # 将所有满足条件的节点设置为sleep状态，供后续进行状态分配
            if nodeSet[node-1].status == 1 or nodeSet[node-1].status == 2:
                nodeSet[node-1].status = 3

    # 3) 先做“图结构合法性”检查：s/t 不在图里就直接失败
    if ('s' not in G) or ('t' not in G):
        print("Graph missing 's' or 't'.")
        return 0, []

    # 4) max-flow 不一定抛异常：不可达时 flow_value=0
    try:
        flow_value, flow_dict = nx.maximum_flow(G, 's', 't')
    except Exception as e:
        print(e)
        print('Can not find the max-flow...')
        return 0, []

    # 5) 显式失败条件：最大流为 0，说明根本没有可用 s->t 路
    if flow_value <= 0:
        print('Max flow is 0: no s->t path.')
        return 0, []

    # 6) 从 flow_dict 里提取路径（你原逻辑保留，但加保护）
    pathList = []
    for key, f in flow_dict.get('s', {}).items():
        if f == 1:
            pathList.append(['s', key])

    # 如果没有从 s 出发的边，直接失败（这在 flow_value>0 理论上不该出现，但稳妥）
    if not pathList:
        print('No outgoing unit-flow edges from s.')
        return 0, []

    for path in pathList:
        # 防止死循环：最多走 |V| 步
        step_limit = len(G.nodes) + 5
        steps = 0
        while path[-1] != 't' and steps < step_limit:
            current = path[-1]
            currentDict = flow_dict.get(current, {})
            moved = False
            for k, f in currentDict.items():
                if f == 1:
                    path.append(k)
                    moved = True
                    break
            if not moved:
                # 这条流“断了”，标记为无效
                break
            steps += 1

    # 7) 选择“最短的有效路径”（必须以 t 结尾才算有效）
    shortestPath = []
    shortestPathLength = 10 ** 9

    for p in pathList:
        if not p or p[-1] != 't':
            continue

        # 去掉 s,t
        inner = p[1:-1]

        # 过滤负节点（更安全：不要在遍历时 remove）
        inner = [x for x in inner if not (isinstance(x, int) and x < 0)]

        if len(inner) > 0 and len(inner) < shortestPathLength:
            shortestPathLength = len(inner)
            shortestPath = inner

    # 8) 再加一道显式判定：没有有效路径就失败
    if not shortestPath:
        print('No valid path extracted from flow decomposition.')
        return 0, []

    print('The selected path in this turn is: ')
    print(shortestPath)

    # 9) 状态更新：先把相关节点标为 active
    for node in shortestPath:
        if node > 0:
            nodeSet[node - 1].status = 1

    # 10) 更新能量
    for node in list(G.nodes):
        if node != 's' and node != 't' and node > 0:
            nodeSet[node - 1].updateEnergy()
            if nodeSet[node - 1].status == 0:
                nodeSet[node - 1].status = 3

    pathFlag = 1
    return pathFlag, shortestPath
