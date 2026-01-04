from itertools import islice

import numpy as np
import matplotlib.pyplot as plt
import math, copy
import networkx as nx
from easydict import EasyDict as edict
import yaml
import random
import pickle
import os

with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = edict(yaml.safe_load(file))

# ===== Q-learning switches =====
# RA级别的 Q-learning
USE_QLEARNING = bool(getattr(config, "use_qlearning", True))   # config.yaml 里可写 use_qlearning: true/false
# 路径调度 Q-learning
USE_PATH_QLEARNING = bool(getattr(config, "use_path_qlearning", True))

# === 路径修复（默认 true）===
USE_PATH_REPAIR = bool(getattr(config, "use_path_repair", True))
MAX_REPAIR_ATTEMPTS = int(getattr(config, "max_repair_attempts", 5))

# === 候选路径数量 K（越大越慢，一般 3~10 足够）===
K_CANDIDATE_PATHS = int(getattr(config, "k_candidate_paths", 5))

# === Q-table 存储 ===
QTABLE_PATH = str(getattr(config, "qtable_path", "qtable_cic.pkl"))
QTABLE_LOAD_ON_START = bool(getattr(config, "qtable_load_on_start", False))
QTABLE_SAVE_EVERY = int(getattr(config, "qtable_save_every", 0))  # 例如 50 表示每 50 轮保存一次；0 表示不保存

# 奖励权衡：能量惩罚系数
Q_LAMBDA_ENERGY = float(getattr(config, "q_lambda_energy", 0.05))
# 路径奖励里可加入路径长度惩罚
Q_LAMBDA_HOPS = float(getattr(config, "q_lambda_hops", 0.0))  # 默认 0 不惩罚

# 全局轮次计数（CICnetworkWithT 每调用一次算一轮）
_GLOBAL_STEP = 0

class QLearningRAAgent:
    """
    对每个 RA 的“选哪一组 nodeSequence 来激活”做 Q-learning。
    Q 表用 dict 存：Q[(state_tuple, action_idx)] = value
    """
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.15, epsilon_min=0.05, epsilon_decay=0.999):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def _q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, valid_actions):
        """epsilon-greedy"""
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        # exploitation
        best_a = valid_actions[0]
        best_q = self._q(state, best_a)
        for a in valid_actions[1:]:
            q = self._q(state, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update(self, state, action, reward, next_state, next_valid_actions):
        """Q(s,a) <- Q + alpha * (r + gamma * max_a' Q(s',a') - Q)"""
        if action is None:
            return
        q_sa = self._q(state, action)
        if next_valid_actions:
            max_next = max(self._q(next_state, a2) for a2 in next_valid_actions)
        else:
            max_next = 0.0
        self.Q[(state, action)] = q_sa + self.alpha * (reward + self.gamma * max_next - q_sa)

    def step_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="qtable_cic.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "Q": self.Q,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            }, f)

    def load(self, path="qtable_cic.pkl"):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.Q = d.get("Q", {})
        self.alpha = d.get("alpha", self.alpha)
        self.gamma = d.get("gamma", self.gamma)
        self.epsilon = d.get("epsilon", self.epsilon)
        self.epsilon_min = d.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = d.get("epsilon_decay", self.epsilon_decay)
        return True

class QLearningPathAgent:
    """
    从 K 条候选路径中选择 1 条路径做“路径调度”。
    action = candidate_path_index (0..K-1)
    Q[(state_tuple, action)] = value
    """
    def __init__(self, alpha=0.2, gamma=0.9, epsilon=0.15, epsilon_min=0.05, epsilon_decay=0.999):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = {}

    def _q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        best_a = valid_actions[0]
        best_q = self._q(state, best_a)
        for a in valid_actions[1:]:
            q = self._q(state, a)
            if q > best_q:
                best_q, best_a = q, a
        return best_a

    def update(self, state, action, reward, next_state, next_valid_actions):
        if action is None:
            return
        q_sa = self._q(state, action)
        if next_valid_actions:
            max_next = max(self._q(next_state, a2) for a2 in next_valid_actions)
        else:
            max_next = 0.0
        self.Q[(state, action)] = q_sa + self.alpha * (reward + self.gamma * max_next - q_sa)

    def step_decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# 全局 agent：保证 main 循环里每一轮调用 CICnetworkWithT 时，Q 表能持续学习
_RA_AGENT = None
_PATH_AGENT = None

def get_ra_agent():
    global _RA_AGENT
    if _RA_AGENT is None:
        _RA_AGENT = QLearningRAAgent(alpha=0.2, gamma=0.9, epsilon=0.15, epsilon_min=0.05, epsilon_decay=0.999)
    return _RA_AGENT

def get_path_agent():
    global _PATH_AGENT
    if _PATH_AGENT is None:
        _PATH_AGENT = QLearningPathAgent(alpha=0.2, gamma=0.9, epsilon=0.15, epsilon_min=0.05, epsilon_decay=0.999)
    return _PATH_AGENT

def save_qtable(path=QTABLE_PATH):
    """
    保存 RA-Q + Path-Q 到同一个文件（向后兼容）
    """
    ra = get_ra_agent()
    pa = get_path_agent()
    payload = {
        "RA": {
            "Q": ra.Q,
            "alpha": ra.alpha, "gamma": ra.gamma, "epsilon": ra.epsilon,
            "epsilon_min": ra.epsilon_min, "epsilon_decay": ra.epsilon_decay,
        },
        "PATH": {
            "Q": pa.Q,
            "alpha": pa.alpha, "gamma": pa.gamma, "epsilon": pa.epsilon,
            "epsilon_min": pa.epsilon_min, "epsilon_decay": pa.epsilon_decay,
        }
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def load_qtable(path=QTABLE_PATH):
    """
    兼容两种格式：
    1) 新格式：{"RA":{...}, "PATH":{...}}
    2) 旧格式：{"Q":..., "alpha":..., ...}（只包含 RA）
    """
    if not os.path.exists(path):
        return False

    with open(path, "rb") as f:
        d = pickle.load(f)

    ra = get_ra_agent()
    pa = get_path_agent()

    # 新格式
    if isinstance(d, dict) and ("RA" in d or "PATH" in d):
        if "RA" in d:
            x = d["RA"]
            ra.Q = x.get("Q", {})
            ra.alpha = x.get("alpha", ra.alpha)
            ra.gamma = x.get("gamma", ra.gamma)
            ra.epsilon = x.get("epsilon", ra.epsilon)
            ra.epsilon_min = x.get("epsilon_min", ra.epsilon_min)
            ra.epsilon_decay = x.get("epsilon_decay", ra.epsilon_decay)
        if "PATH" in d:
            x = d["PATH"]
            pa.Q = x.get("Q", {})
            pa.alpha = x.get("alpha", pa.alpha)
            pa.gamma = x.get("gamma", pa.gamma)
            pa.epsilon = x.get("epsilon", pa.epsilon)
            pa.epsilon_min = x.get("epsilon_min", pa.epsilon_min)
            pa.epsilon_decay = x.get("epsilon_decay", pa.epsilon_decay)
        return True

    # 旧格式（只装 RA）
    if isinstance(d, dict) and "Q" in d:
        ra.Q = d.get("Q", {})
        ra.alpha = d.get("alpha", ra.alpha)
        ra.gamma = d.get("gamma", ra.gamma)
        ra.epsilon = d.get("epsilon", ra.epsilon)
        ra.epsilon_min = d.get("epsilon_min", ra.epsilon_min)
        ra.epsilon_decay = d.get("epsilon_decay", ra.epsilon_decay)
        return True

    return False

# ============================================================
# 2) 状态离散化：RA state & Path state
# ============================================================
def ra_state(ra, energy_threshold):
    """
    把连续状态离散化成 tuple，便于 Q 表索引。
    你现在最稳定/最有用的特征是：
    - 可用组数（=len(nodeSequence)）
    - RA 内可用节点平均能量 / energy_threshold 的比值
    - fail 比例
    """
    usable = [n for n in ra.nodes if n.status != 4 and n.Energy >= energy_threshold]
    fail_cnt = len([n for n in ra.nodes if n.status == 4])

    group_cnt = len(ra.nodeSequence)
    # 能量比值离散化
    if usable:
        avg_e = sum(n.Energy for n in usable) / len(usable)
        ratio = avg_e / max(energy_threshold, 1e-6)
    else:
        ratio = 0.0

    # 分箱
    if ratio < 1.0: e_bin = 0
    elif ratio < 2.0: e_bin = 1
    elif ratio < 3.0: e_bin = 2
    else: e_bin = 3

    # group_cnt 分箱
    if group_cnt <= 0: g_bin = 0
    elif group_cnt == 1: g_bin = 1
    elif group_cnt == 2: g_bin = 2
    else: g_bin = 3  # 3 表示 3+

    # fail 比例分箱
    if len(ra.nodes) == 0:
        f_bin = 0
    else:
        fr = fail_cnt / len(ra.nodes)
        if fr < 0.1: f_bin = 0
        elif fr < 0.3: f_bin = 1
        elif fr < 0.6: f_bin = 2
        else: f_bin = 3

    return (g_bin, e_bin, f_bin)

def path_state(candidate_paths, RA, energy_threshold):
    """
    路径调度用 state（离散化）
    这里给一个“够用且稳定”的 state：
    - 候选路径数量 K_bin
    - 最短候选路径长度 L_bin
    - 候选路径整体能量水平 E_bin（取最短路径上可用节点平均能量比）
    """
    k = len(candidate_paths)
    if k <= 0:
        return (0, 0, 0)

    lengths = [len(p) for p in candidate_paths]
    Lmin = min(lengths)

    # K bin
    if k <= 1: k_bin = 0
    elif k <= 3: k_bin = 1
    elif k <= 6: k_bin = 2
    else: k_bin = 3

    # L bin
    if Lmin <= 4: L_bin = 0
    elif Lmin <= 6: L_bin = 1
    elif Lmin <= 9: L_bin = 2
    else: L_bin = 3

    # energy bin：看最短路径上，各 RA 内“可用节点平均能量 / threshold”
    best_path = candidate_paths[int(np.argmin(lengths))]
    ratios = []
    for ra_id in best_path:
        ra = RA[ra_id]
        usable = [n for n in ra.nodes if n.status != 4 and n.Energy >= energy_threshold]
        if usable:
            avg_e = sum(n.Energy for n in usable) / len(usable)
            ratios.append(avg_e / max(energy_threshold, 1e-6))
    r = float(np.mean(ratios)) if len(ratios) else 0.0

    if r < 1.0: E_bin = 0
    elif r < 2.0: E_bin = 1
    elif r < 3.0: E_bin = 2
    else: E_bin = 3

    return (k_bin, L_bin, E_bin)
class reconstructionArea(object):
    def __init__(self, centerPointLocation):
        self.nodes = []                # 真实 Node 引用
        self.centerPointLocation = centerPointLocation
        self.weight = 0
        self.nodeSequence = []

    def addNode(self, node):
        self.nodes.append(node)

    def active_nodes(self):
        return [n for n in self.nodes if n.status == 1]  # 1=ACTIVE

    def satisfy_with(self, nodes):
        if not nodes:
            return False
        fai = RMSE(self.centerPointLocation[0], self.centerPointLocation[1], nodes)
        return fai < config.sigma

    def recompute_groups_and_weight(self, energy_threshold):
        """
        用“当前可用节点”动态构造 nodeSequence:
        - 单点可满足 sigma -> 单点组
        - 剩余点协作可满足 sigma -> 协作组
        """
        self.nodeSequence = []
        usable = [n for n in self.nodes if (n.status != 4) and (n.Energy >= energy_threshold)]
        cnt = 0
        rest = []

        for n in usable:
            if self.satisfy_with([n]):
                cnt += 1
                self.nodeSequence.append([n])
            else:
                rest.append(n)

        if rest and self.satisfy_with(rest):
            cnt += 1
            self.nodeSequence.append(rest)

        self.weight = cnt

def generateGraph(nodeSet, energy_threshold=None):
    """
    优化点：
    - RA 节点仅连 8 邻域，不再 O(N^2) 两重循环
    - 每轮根据节点能量/失败状态动态计算 RA.weight 和 nodeSequence
    """
    if energy_threshold is None:
        energy_threshold = config.energyThreshold

    bias = config.CR / 2
    X = math.ceil(config.map[0] / config.CR)  # 行数（沿 map[0]）
    Y = math.ceil(config.map[1] / config.CR)  # 列数（沿 map[1]）

    RA = []
    for i in range(X):
        x = i * config.CR + bias
        for j in range(Y):
            y = j * config.CR + bias
            RA.append(reconstructionArea(centerPointLocation=[x, y]))

    # 分配节点到 RA
    for n in nodeSet:
        gx = math.ceil(n.location[0] / config.CR)  # 1..X
        gy = math.ceil(n.location[1] / config.CR)  # 1..Y
        gx = min(max(gx, 1), X)
        gy = min(max(gy, 1), Y)
        idx = calculateRAIndex(gx, gy)
        RA[idx].addNode(n)

    # 计算每个 RA 的可用组与点权
    weights = [0] * len(RA)
    for i in range(len(RA)):
        RA[i].recompute_groups_and_weight(energy_threshold)
        weights[i] = RA[i].weight

    # 构图：只加入 weight>0 的 RA
    G = nx.Graph()
    for i in range(len(RA)):
        if weights[i] > 0:
            G.add_node(i)

    # 8 邻域连边（线性复杂度）
    def neighbors(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < X and 0 <= cc < Y:
                    yield rr, cc

    for r in range(X):
        for c in range(Y):
            i = r * Y + c
            if i not in G:
                continue
            for rr, cc in neighbors(r, c):
                j = rr * Y + cc
                if j in G:
                    G.add_edge(i, j)

    # 虚拟源汇
    G.add_node('s')
    G.add_node('t')

    # 上边界行（r=0）连 s，下边界行（r=X-1）连 t
    for c in range(Y):
        top = 0 * Y + c
        if top in G:
            G.add_edge('s', top)
        bottom = (X-1) * Y + c
        if bottom in G:
            G.add_edge(bottom, 't')

    return G, weights, RA

def calculateRAIndex(x, y):
    index = (x-1) * (math.ceil(config.map[1]/config.CR)) + y-1
    return index

def edgeExist(i, j):
    unitX = math.ceil(config.map[1] / config.CR)
    unitY = math.ceil(config.map[0] / config.CR)
    xi = i // unitX
    yi = i % unitY
    xj = j // unitX
    yj = j % unitY
    # print(i,':(',xi,", ",yi,")")

    disX = math.fabs(xi - xj)
    disY = math.fabs(yi - yj)
    if disX <= 1 and disY <= 1:
        return True
    else:
        return False

def RMSE(centerX, centerY, nodeSet):
    Variogram_N00_a1 = np.zeros((len(nodeSet)+1, 1))
    H_a1 = np.zeros((len(nodeSet)+1, 1))

    # 计算y(si, x)
    i = 0
    for n in nodeSet:
        H_i0_a1 = math.sqrt(pow((n.location[0] - centerX), 2) + pow((n.location[1] - centerY), 2))       # 领域内节点与原点之间的距离
        H_a1[i][0] = H_i0_a1
        if H_i0_a1 == 0:
            V_i0_a1 = H_i0_a1
        else:
            V_i0_a1 = 1 - math.exp(-(pow(H_i0_a1,2)/ pow(config.CR,2)))                             #  根据标准高斯变差函数计算变差值
        Variogram_N00_a1[i][0] = V_i0_a1
        i = i + 1
    # end
    Variogram_N00_a1[len(nodeSet)][0] = 1
    Variogram_N00_a1_mat = np.mat(Variogram_N00_a1)


    # 计算y(si, sj)
    Variogram_Njj1_a1 = np.zeros((len(nodeSet)+1, len(nodeSet)+1))
    for i in range(0, len(nodeSet)):
        for j in range(0, len(nodeSet)):
            H_Nj_a1 = math.sqrt(pow((nodeSet[i].location[0] - nodeSet[j].location[0]), 2) + pow((nodeSet[i].location[1] - nodeSet[j].location[1]), 2))
            if H_Nj_a1 == 0:
                V_Nj_a1 = H_Nj_a1
                Variogram_Njj1_a1[i][j] = V_Nj_a1
            else:
                # print(i,j)
                V_Nj_a1 = 1 - math.exp(-(pow(H_Nj_a1, 2) / pow(config.CR, 2)))
                # print('V_Nj_a1: ', V_Nj_a1)
                Variogram_Njj1_a1[i][j] = V_Nj_a1

    for i in range(0, len(nodeSet)):
        Variogram_Njj1_a1[len(nodeSet)][i] = 1
        Variogram_Njj1_a1[i][len(nodeSet)] = 1
    Variogram_Njj1_a1[len(nodeSet)][len(nodeSet)] = 0
    # end


    Variogram_Njj1_a1_mat = np.mat(Variogram_Njj1_a1)
    Variogram_Njj1_a1_mat = Variogram_Njj1_a1_mat.I
    Variance0_a1 = np.dot(Variogram_Njj1_a1_mat, Variogram_N00_a1_mat)
    total = 0
    for i in range(0, len(nodeSet)+1):
        total = total + Variance0_a1[i] * Variogram_N00_a1[i]
    Fai = math.sqrt(total)
    return Fai

def get_candidate_paths(G, K):
    """
    路径候选生成：返回 K 条候选路径（去掉 s/t，仅保留 RA id 序列）
    """
    if not (('s' in G) and ('t' in G)):
        return []
    try:
        gen = nx.shortest_simple_paths(G, 's', 't')  # 依路径长度从短到长
        paths = []
        for p in islice(gen, K):
            ra_ids = p[1:-1]
            paths.append(ra_ids)
        return paths
    except Exception:
        return []

# 这是只考虑是否可以形成栅栏的简单函数
def CICnetwork(nodeSet):
    # step1: Initialization 生成节点，原始节点构成原始图
    G, weights, RA = generateGraph(nodeSet)
    pathNodeList = []
    pathCount = 0

    while 1:
        try:
            nodeList = nx.dijkstra_path(G, 's', 't')  # 最短路径
            print('The activated path is :', nodeList)
            pathNodeList.append(nodeList)
            nodeInThisTurn = nodeList[1:len(nodeList) - 1]
            # 更新RA-weight权重,若权重为0，则删除这个节点
            minWeight = config.nodeNumber
            for element in nodeInThisTurn:
                if weights[element] < minWeight:
                    minWeight = weights[element]
            for element in nodeInThisTurn:
                weights[element] = weights[element] - minWeight
                if weights[element] <= 0:
                    G.remove_node(element)
            pathCount = pathCount + minWeight

        except:
            print('Can not find a path anymore...')
            print('The total pathCount:', pathCount)
            break
    return pathCount, pathNodeList

    # step2: Malicious node detection
        # Can the node justify if it is intruded by itself or not?

    # step3: Status update
    # for node in nodeList:
    #     node.updateEnergy()

def CICnetworkWithT(nodeSet, intrudedNodes=None):
    """
    intrudedNodes 统一约定为 Node.ID（1-based）列表。
    在本函数中统一用 nodeSet[id-1] 定位节点。

    返回：
      (pathFlag, nodeInThisTurn)
      pathFlag=1：本轮成功形成 barrier 路径
      pathFlag=0：失败（无路/无法在某些 RA 选出可用组）
    """
    global _GLOBAL_STEP
    _GLOBAL_STEP += 1

    # --- 载入 Q-table（只在启动时做一次）---
    if _GLOBAL_STEP == 1 and QTABLE_LOAD_ON_START:
        ok = load_qtable(QTABLE_PATH)
        if ok:
            ra = get_ra_agent()
            pa = get_path_agent()
            print(f"[CIC-Q] Loaded Q-table from {QTABLE_PATH}. RA={len(ra.Q)} PATH={len(pa.Q)}")
        else:
            print(f"[CIC-Q] No existing Q-table at {QTABLE_PATH}, start fresh.")

    ra_agent = get_ra_agent()
    path_agent = get_path_agent()

    # (1) 标记入侵/失效节点（可选）
    if intrudedNodes:
        for node_id in set(intrudedNodes):
            if isinstance(node_id, int):
                idx = node_id - 1
                if 0 <= idx < len(nodeSet):
                    nodeSet[idx].status = 4  # failed

    # (2) 能量阈值淘汰
    for n in nodeSet:
        if n.status != 4 and n.Energy < config.energyThreshold:
            n.status = 4

    # (3) 构图
    G, weights, RA = generateGraph(nodeSet, energy_threshold=config.energyThreshold)

    # ========================================================
    # (4) 路径调度 + 路径修复（循环尝试）
    #     blocked_ras：本轮修复时临时阻塞的 RA（不改变 nodeSet）
    # ========================================================
    blocked_ras = set()
    chosen_path = None
    chosen_path_action = None
    path_state_s = None
    used_repair_times = 0

    # 我们先不“提交”激活动作，先做 planning 成功后再 commit
    planned_actions = None  # list of dict: {ra_id, group, state, action, pre_energy_sum, valid_actions}

    for attempt in range(MAX_REPAIR_ATTEMPTS if USE_PATH_REPAIR else 1):
        # 复制图并阻塞（用于修复）
        G_try = G.copy()
        for b in blocked_ras:
            if b in G_try:
                G_try.remove_node(b)

        # 生成 K 条候选路径
        candidates = get_candidate_paths(G_try, K_CANDIDATE_PATHS)
        if not candidates:
            # 没路了
            chosen_path = None
            break

        # 选择路径（RL 或 baseline）
        valid_path_actions = list(range(len(candidates)))
        if USE_PATH_QLEARNING:
            path_state_s = path_state(candidates, RA, config.energyThreshold)
            chosen_path_action = path_agent.choose_action(path_state_s, valid_path_actions)
            if chosen_path_action is None:
                chosen_path_action = 0
        else:
            path_state_s = None
            chosen_path_action = 0  # baseline 选最短的第一条

        chosen_path = candidates[chosen_path_action]

        # ====================================================
        # 对该路径做 “RA 组选择 planning”
        # 任何 RA 选不出可用组 -> 触发修复（阻塞该 RA 重来）
        # ====================================================
        tmp_plan = []
        planning_failed_ra = None

        for ra_id in chosen_path:
            ra = RA[ra_id]

            # 计算 valid_actions（注意：planning 阶段不要改 node.status，避免污染）
            valid_actions = []
            for a_idx, group in enumerate(ra.nodeSequence):
                ok = True
                for n in group:
                    if n.status == 4 or n.Energy < config.energyThreshold:
                        ok = False
                        break
                if ok:
                    valid_actions.append(a_idx)

            if not valid_actions:
                planning_failed_ra = ra_id
                break

            # 选择组：RA-Q 或 baseline
            if USE_QLEARNING:
                s = ra_state(ra, config.energyThreshold)
                a = ra_agent.choose_action(s, valid_actions)
                if a is None:
                    a = valid_actions[0]
            else:
                s = None
                a = valid_actions[0]

            group = ra.nodeSequence[a]
            pre_energy_sum = sum(n.Energy for n in group)

            tmp_plan.append({
                "ra_id": ra_id,
                "state": s,
                "action": a,
                "group": group,
                "valid_actions": valid_actions,
                "pre_energy_sum": pre_energy_sum,
            })

        if planning_failed_ra is None:
            # planning 成功
            planned_actions = tmp_plan
            break

        # planning 失败：修复
        used_repair_times += 1
        if not USE_PATH_REPAIR:
            chosen_path = None
            planned_actions = None
            break
        blocked_ras.add(planning_failed_ra)

    # 如果最终没成功 plan 出一条完整可执行路径
    if not planned_actions or chosen_path is None:
        print("Can not find a path anymore...")
        return 0, []

    # ========================================================
    # (5) Commit：按计划真正设置状态（sleep -> active）
    # ========================================================
    for step in planned_actions:
        ra_id = step["ra_id"]
        ra = RA[ra_id]

        # 先清理：RA 内非 failed -> sleep
        for n in ra.nodes:
            if n.status != 4:
                n.status = 3

        # 再激活 chosen group
        for n in step["group"]:
            n.status = 1

    print("The activated path is :", ['s'] + chosen_path + ['t'])

    # (6) 更新能量
    total_energy_cost = 0.0
    for step in planned_actions:
        pre = step["pre_energy_sum"]
        # updateEnergy 后再算 post
        # 注意：这里 post 必须在 updateEnergy 之后才能准确，所以先累计 pre，post 后面补
        step["_pre_sum"] = pre

    for n in nodeSet:
        n.updateEnergy()
        if n.status == 0:
            n.status = 3

    for step in planned_actions:
        post = sum(n.Energy for n in step["group"])
        total_energy_cost += max(0.0, step["_pre_sum"] - post)

# (7) Q-learning 更新（仅开启时）
#     1) RA-Q：每个 RA 一条 transition
#     2) Path-Q：整条路径一条 transition（调度 + 修复）
    # ---- RA-Q update ----
    if USE_QLEARNING:
        for step in planned_actions:
            ra_id = step["ra_id"]
            ra = RA[ra_id]

            # 覆盖判定（你原逻辑：当前 RA 内 active 节点是否满足 sigma）
            active = [n for n in ra.nodes if n.status == 1]
            cover_ok = ra.satisfy_with(active)

            # reward：覆盖成功 +1，失败 -2；再减能量惩罚
            # （能量惩罚用该 RA 组的能量差，而不是总能量差）
            pre = step["pre_energy_sum"]
            post = sum(n.Energy for n in step["group"])
            energy_cost = max(0.0, pre - post)

            reward = (1.0 if cover_ok else -2.0) - Q_LAMBDA_ENERGY * energy_cost

            s = step["state"]
            a = step["action"]
            s2 = ra_state(ra, config.energyThreshold)

            # next_valid_actions：用当前 ra.nodeSequence 重新计算
            next_valid = []
            for a_idx, group in enumerate(ra.nodeSequence):
                ok = True
                for n in group:
                    if n.status == 4 or n.Energy < config.energyThreshold:
                        ok = False
                        break
                if ok:
                    next_valid.append(a_idx)

            ra_agent.update(s, a, reward, s2, next_valid)

        ra_agent.step_decay()

    # ---- Path-Q update ----
    if USE_PATH_QLEARNING:
        # path reward：本轮成功 +1，减能量、减 hop、修复次数惩罚
        # 你也可以把 “修复次数”当作稳定性惩罚
        path_len = len(chosen_path)
        reward_path = 1.0 - Q_LAMBDA_ENERGY * total_energy_cost - Q_LAMBDA_HOPS * float(path_len) - 0.1 * float(
            used_repair_times)

        # next_state：下轮才真正知道；这里用当前 G 的候选作为近似 next_state（足够稳定）
        candidates2 = get_candidate_paths(G, K_CANDIDATE_PATHS)
        s2 = path_state(candidates2, RA, config.energyThreshold)
        next_valid = list(range(len(candidates2))) if len(candidates2) > 0 else []

        path_agent.update(path_state_s, chosen_path_action, reward_path, s2, next_valid)
        path_agent.step_decay()

    # ========================================================
    # (8) 定期保存 Q-table（RA + PATH）
    # ========================================================
    if QTABLE_SAVE_EVERY > 0 and (_GLOBAL_STEP % QTABLE_SAVE_EVERY == 0):
        save_qtable(QTABLE_PATH)
        ra = get_ra_agent()
        pa = get_path_agent()
        print(f"[CIC-Q] Saved Q-table to {QTABLE_PATH} at step={_GLOBAL_STEP}, RA={len(ra.Q)} PATH={len(pa.Q)}")

    return 1, chosen_path