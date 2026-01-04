import math
from easydict import EasyDict as edict
import yaml


with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = edict(yaml.safe_load(file))

# 节点类
class Node(object):
    # 类成员，方法，数据属性
    count = 0

    # 统一状态语义
    UNDEFINED = 0
    ACTIVE = 1     # functional (sense + tx + rx)
    RELAY = 2      # relay (tx + rx)
    SLEEP = 3      # sleep/standby (no cost)
    FAILED = 4     # failed (no cost)
    # OUT_OF_ENERGY = 5  # 如果需要再加

    def __init__(self, location, sensingRange, transmissionRange, initialEnergy):
        self.status = Node.UNDEFINED
        self.isSelect = False                           # 记录当前是否被选择为簇头
        self.isVisited = False
        self.isActivated = False                        # 记录当前节点是否被激活，默认处于休眠状态
        self.location = location
        self.ID = Node.count + 1
        self.sensingRange = sensingRange
        self.transmissionRange = transmissionRange
        self.Energy = initialEnergy                     # 记录该节点当前能量值，初始化为initialEnergy
        self.isRedundancy = bool(False)                 # 记录该节点是否为冗余节点
        self.redundancySet = []                         # 每个节点都有自己的冗余集，即与自己重合的节点，当isRedundancy为False的时候才会引用该值
        Node.count += 1

    def resetCount(self):
        Node.count = 0

    def clusterHead(self, clusterHead):                 # 节点当前的簇头（开始是虚拟的，后续是真实的）
        self.clusterHead = clusterHead

    def distanceWithNode(self, anotherNode):
        # print(self.location)
        # print(anotherNode.location)
        distance = math.sqrt(math.pow(self.location[0]-anotherNode.location[0], 2) + math.pow(self.location[1]-anotherNode.location[1], 2) )
        # print("distance: ", distance)
        return distance

    def resEnergy(self, T):
        avgEnergy = (config.Et + config.Er + config.Es) * T
        resEnergy = self.Energy - avgEnergy
        return resEnergy

    def updateEnergy(self, d=1, k=100000):
        if self.status == Node.UNDEFINED:
            # 不建议在仿真循环里出现
            return self.Energy

        # Sensing
        Esx = config.Es if self.status == Node.ACTIVE else 0.0

        # Tx/Rx
        if self.status in (Node.ACTIVE, Node.RELAY):
            if d < config.d0:
                Etx = k * config.Eelec + k * config.epsilonfs * (d ** 2)
            else:
                # 你原来这里仍用 epsilonfs，我建议用 epsilonmp（如果 config 里有）
                eps = getattr(config, "epsilonmp", config.epsilonfs)
                Etx = k * config.Eelec + k * eps * (d ** 4)
            Erx = k * config.Eelec
            self.Energy -= (Etx + Erx + Esx)
            self.Energy = max(0.0, self.Energy)

        # SLEEP/FAILED 不耗能（或只耗极低常量，可自行加）
        return self.Energy
