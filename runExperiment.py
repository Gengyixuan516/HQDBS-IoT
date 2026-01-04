# -*- coding: utf-8 -*-
import os
import random, time, copy, csv, yaml
import itertools,contextlib, sys, io
from typing import Dict, Any, List, Tuple

import numpy as np
from collections import defaultdict
from easydict import EasyDict as edict

import networkModel, MSPA, MaxFlow, fuzzyGraph
from DRLCIC import CICnetwork


@contextlib.contextmanager
def suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err
# ---------------------------
# Config
# ---------------------------
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8-sig") as f:
        return edict(yaml.safe_load(f))


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def patch_module_config(mod, cfg_updates: Dict[str, Any]):
    """把 main 里 cfg 的关键字段同步写入各模块的 module.config（如果存在）"""
    mc = getattr(mod, "config", None)
    if mc is None:
        return
    for k, v in cfg_updates.items():
        try:
            setattr(mc, k, v)
        except Exception:
            pass


def patch_all_configs(cfg: edict):
    """保证所有模块对齐同一份实验参数（尤其 nodeNumber/T/energyThreshold/...）"""
    updates = {
        "nodeNumber": int(cfg.nodeNumber),
        "T": int(cfg.T),
        "energyThreshold": float(cfg.energyThreshold),
        "use_qlearning": bool(getattr(cfg, "use_qlearning", False)),
        "use_path_qlearning": bool(getattr(cfg, "use_path_qlearning", False)),
        "qtable_load_on_start": bool(getattr(cfg, "qtable_load_on_start", False)),
        "qtable_save_every": int(getattr(cfg, "qtable_save_every", 0)),
        "qtable_path": str(getattr(cfg, "qtable_path", "qtable_cic.pkl")),
        "q_lambda_energy": float(getattr(cfg, "q_lambda_energy", 0.05)),
        "use_dynamic_weight_graph": bool(getattr(cfg, "use_dynamic_weight_graph", True)),
    }
    for m in [networkModel, MSPA, MaxFlow, fuzzyGraph, CICnetwork]:
        patch_module_config(m, updates)

    # 同时把 CICnetwork 的“运行时开关”也同步（如果你在 CICnetwork.py 里用了全局变量）
    if hasattr(CICnetwork, "USE_QLEARNING"):
        CICnetwork.USE_QLEARNING = bool(getattr(cfg, "use_qlearning", False))
    if hasattr(CICnetwork, "USE_DYNAMIC_WEIGHT_GRAPH"):
        CICnetwork.USE_DYNAMIC_WEIGHT_GRAPH = bool(getattr(cfg, "use_dynamic_weight_graph", True))
    if hasattr(CICnetwork, "QTABLE_SAVE_EVERY"):
        CICnetwork.QTABLE_SAVE_EVERY = int(getattr(cfg, "qtable_save_every", 0))
    if hasattr(CICnetwork, "QTABLE_PATH"):
        CICnetwork.QTABLE_PATH = str(getattr(cfg, "qtable_path", "qtable_cic.pkl"))


# ---------------------------
# Intrusion (two-stage)
# ---------------------------
def sample_intrusions_two_stage(nodeSet: list, p_slot: float, p_node: float) -> List[int]:
    """
    二段式入侵：
      1) 本 time slot 发生攻击概率 p_slot
      2) 若发生攻击，每节点被攻击概率 p_node
    返回：Node.ID (1-based)
    """
    intruded_ids = []
    if p_slot <= 0 or p_node <= 0:
        return intruded_ids
    if random.random() >= p_slot:
        return intruded_ids

    for n in nodeSet:
        if random.random() < p_node:
            n.status = 4
            intruded_ids.append(n.ID)
    return intruded_ids


def apply_intrusions_by_id(nodeSet: list, intruded_ids: List[int]):
    """把同一批 Node.ID 的节点标记为 failed（1-based）"""
    if not intruded_ids:
        return
    for node_id in set(intruded_ids):
        idx = node_id - 1
        if 0 <= idx < len(nodeSet):
            nodeSet[idx].status = 4


# ---------------------------
# Energy stats
# ---------------------------
def capture_initial_energy(nodeSet: list) -> Dict[int, float]:
    return {n.ID: float(n.Energy) for n in nodeSet}


def energy_consumption_variance(nodeSet: list, e0: Dict[int, float]) -> Tuple[float, float]:
    """
    返回 (mean_consumption, var_consumption)，消费=E0-Et
    """
    cons = []
    for n in nodeSet:
        e_init = e0.get(n.ID, float(n.Energy))
        cons.append(float(e_init) - float(n.Energy))
    if not cons:
        return 0.0, 0.0
    mean = sum(cons) / len(cons)
    var = sum((x - mean) ** 2 for x in cons) / len(cons)
    return mean, var

def calc_total_consumption_and_power(nodeSet, e0_map, t_fail, overall_stop_t):
    # t_run：失败则用 fail-1；不失败用 overall_stop_t
    if t_fail is None:
        t_run = int(overall_stop_t)
    else:
        t_run = max(1, int(t_fail) - 1)

    # 总能耗
    cons = 0.0
    for n in nodeSet:
        e0 = float(e0_map.get(n.ID, 0.0))
        cons += max(0.0, e0 - float(n.Energy))

    p_avg = cons / float(t_run)
    return cons, p_avg, t_run

# ---------------------------
# Static experiment (no time)
# ---------------------------
def run_static_barrier_counts(cfg: edict, nodeSet: list) -> Dict[str, Any]:
    res = {"MSPA_pathCount": None, "MaxFlow_pathCount": None, "EC2_pathCount": None, "CIC_pathCount": None}

    if cfg.enable_MSPA:
        try:
            ns = copy.deepcopy(nodeSet)
            pc, paths = MSPA.MSPA(ns)
            res["MSPA_pathCount"] = float(pc)
            res["MSPA_pathNum"] = len(paths) if paths is not None else None
        except Exception:
            res["MSPA_pathCount"] = None

    if cfg.enable_MaxFlow:
        try:
            ns = copy.deepcopy(nodeSet)
            pc, paths = MaxFlow.MaxFlow(ns)
            res["MaxFlow_pathCount"] = float(pc)
            res["MaxFlow_pathNum"] = len(paths) if paths is not None else None
        except Exception:
            res["MaxFlow_pathCount"] = None

    if cfg.enable_EC2:
        try:
            ns = copy.deepcopy(nodeSet)
            pc, paths = fuzzyGraph.EC2(ns)
            res["EC2_pathCount"] = float(pc)
            res["EC2_pathNum"] = len(paths) if paths is not None else None
        except Exception:
            res["EC2_pathCount"] = None

    if cfg.enable_CIC:
        try:
            ns = copy.deepcopy(nodeSet)
            pc, paths = CICnetwork.CICnetwork(ns)
            res["CIC_pathCount"] = float(pc)
            res["CIC_pathNum"] = len(paths) if paths is not None else None
        except Exception:
            res["CIC_pathCount"] = None

    return res


# ---------------------------
# Lifetime simulation (with time)
# ---------------------------
def run_lifetime_simulation(cfg: edict, base_nodeSet: list) -> Dict[str, Any]:
    """
    输出：
      - 每种方法第一次 fail 的时间 t_fail（从1开始，若一直不fail则为 None）
      - overall_stop_t: 所有方法都 fail 或到 T 的终止时间
      - 能量消耗方差（可选）
    """
    # === 可选：每次 run 重置 CIC 的 RL 状态（避免跨 seed 污染）===
    if bool(getattr(cfg, "reset_cic_each_run", False)):
        try:
            CICnetwork.reset_rl()  # 你需要在 CICnetwork.py 里提供 reset_rl()
        except Exception:
            pass

    # copies
    ns_mspa = copy.deepcopy(base_nodeSet)
    ns_mf   = copy.deepcopy(base_nodeSet)
    ns_ec2  = copy.deepcopy(base_nodeSet)
    ns_cic  = copy.deepcopy(base_nodeSet)

    # init graphs（多数 WithT 需要）
    G_mspa = G_mf = G_ec2 = None
    if cfg.enable_MSPA:
        try:
            G_mspa, _ = MSPA.generateGraph(ns_mspa)
        except Exception:
            G_mspa = None
    if cfg.enable_MaxFlow:
        try:
            G_mf = MaxFlow.generateGraph(ns_mf)
        except Exception:
            G_mf = None
    if cfg.enable_EC2:
        try:
            G_ec2 = fuzzyGraph.generateGraph(ns_ec2)
        except Exception:
            G_ec2 = None

    status = {
        "MSPA": 1 if cfg.enable_MSPA else 0,
        "MaxFlow": 1 if cfg.enable_MaxFlow else 0,
        "EC2": 1 if cfg.enable_EC2 else 0,
        "CIC": 1 if cfg.enable_CIC else 0,
    }
    t_fail = {k: None for k in status.keys()}

    # energy baseline
    e0_mspa = capture_initial_energy(ns_mspa)
    e0_mf   = capture_initial_energy(ns_mf)
    e0_ec2  = capture_initial_energy(ns_ec2)
    e0_cic  = capture_initial_energy(ns_cic)

    overall_stop_t = 0

    for t in range(1, int(cfg.T) + 1):
        overall_stop_t = t
        print('********************************* t =', t, '*********************************')

        # intrusion IDs sampled from a reference set（按 Node.ID 传播到各 copies）
        intruded_ids = []
        if bool(getattr(cfg, "enable_intrusion", False)):
            p_slot = float(getattr(cfg, "p_slot", getattr(cfg, "pIntrusion", 0.0)))
            p_node = float(getattr(cfg, "p_node", 0.05))
            intruded_ids = sample_intrusions_two_stage(base_nodeSet, p_slot, p_node)

            apply_intrusions_by_id(ns_mspa, intruded_ids)
            apply_intrusions_by_id(ns_mf, intruded_ids)
            apply_intrusions_by_id(ns_ec2, intruded_ids)
            apply_intrusions_by_id(ns_cic, intruded_ids)

        # MSPA
        if status["MSPA"]:
            try:
                print('------------------MSPA--------------------')
                status["MSPA"], _ = MSPA.MSPAWithT(ns_mspa, G_mspa, intruded_ids)
            except Exception:
                status["MSPA"] = 0
            if status["MSPA"] == 0 and t_fail["MSPA"] is None:
                t_fail["MSPA"] = t

        # MaxFlow
        if status["MaxFlow"]:
            try:
                print('------------------Maxflow--------------------')
                status["MaxFlow"], _ = MaxFlow.MaxFlowWithT(ns_mf, G_mf, intruded_ids)
            except Exception:
                status["MaxFlow"] = 0
            if status["MaxFlow"] == 0 and t_fail["MaxFlow"] is None:
                t_fail["MaxFlow"] = t

        # EC2
        if status["EC2"]:
            try:
                print('------------------EC2--------------------')
                status["EC2"], _ = fuzzyGraph.EC2WithT(ns_ec2, G_ec2, intruded_ids)
            except Exception:
                status["EC2"] = 0
            if status["EC2"] == 0 and t_fail["EC2"] is None:
                t_fail["EC2"] = t

        # CIC
        if status["CIC"]:
            try:
                print('------------------CICNet--------------------')
                status["CIC"], _ = CICnetwork.CICnetworkWithT(ns_cic, intruded_ids)
            except Exception:
                status["CIC"] = 0
            if status["CIC"] == 0 and t_fail["CIC"] is None:
                t_fail["CIC"] = t

        # stop when all fail
        if bool(getattr(cfg, "stop_when_all_fail", True)):
            if not (status["MSPA"] or status["MaxFlow"] or status["EC2"] or status["CIC"]):
                # === 新增：全部 fail 时保存一次最终 Q 表（可选开关）===
                if bool(getattr(cfg, "save_final_qtable", False)):
                    try:
                        # CICnetwork.py 里我给的是 save_qtable(...)；如果你封装成 agent.save(...)，改成对应调用即可
                        CICnetwork.save_qtable(getattr(cfg, "qtable_path", "qtable_cic.pkl"))
                        print("[CIC-Q] saved FINAL Q-table on all-fail")
                    except Exception as e:
                        print("[CIC-Q] final save failed:", e)

                break

    # energy variance（实验5会用）
    mean_mspa, var_mspa = energy_consumption_variance(ns_mspa, e0_mspa)
    mean_mf,   var_mf   = energy_consumption_variance(ns_mf,   e0_mf)
    mean_ec2,  var_ec2  = energy_consumption_variance(ns_ec2,  e0_ec2)
    mean_cic,  var_cic  = energy_consumption_variance(ns_cic,  e0_cic)
    # total consumption & avg power(Exp7)
    cons_mspa, p_mspa, t_run_mspa = calc_total_consumption_and_power(ns_mspa, e0_mspa, t_fail["MSPA"], overall_stop_t)
    cons_mf, p_mf, t_run_mf = calc_total_consumption_and_power(ns_mf, e0_mf, t_fail["MaxFlow"], overall_stop_t)
    cons_ec2, p_ec2, t_run_ec2 = calc_total_consumption_and_power(ns_ec2, e0_ec2, t_fail["EC2"], overall_stop_t)
    cons_cic, p_cic, t_run_cic = calc_total_consumption_and_power(ns_cic, e0_cic, t_fail["CIC"], overall_stop_t)

    return {
        "t_fail_MSPA": t_fail["MSPA"],
        "t_fail_MaxFlow": t_fail["MaxFlow"],
        "t_fail_EC2": t_fail["EC2"],
        "t_fail_CIC": t_fail["CIC"],
        "overall_stop_t": overall_stop_t,

        "energy_mean_MSPA": mean_mspa,
        "energy_var_MSPA": var_mspa,
        "energy_mean_MaxFlow": mean_mf,
        "energy_var_MaxFlow": var_mf,
        "energy_mean_EC2": mean_ec2,
        "energy_var_EC2": var_ec2,
        "energy_mean_CIC": mean_cic,
        "energy_var_CIC": var_cic,

        "energy_cons_MSPA": cons_mspa, "avg_power_MSPA": p_mspa, "t_run_MSPA": t_run_mspa,
        "energy_cons_MaxFlow": cons_mf, "avg_power_MaxFlow": p_mf, "t_run_MaxFlow": t_run_mf,
        "energy_cons_EC2": cons_ec2, "avg_power_EC2": p_ec2, "t_run_EC2": t_run_ec2,
        "energy_cons_CIC": cons_cic, "avg_power_CIC": p_cic, "t_run_CIC": t_run_cic,
    }


# ---------------------------
# timecost (with time)
# ---------------------------
def build_intrusion_schedule(base_nodeSet, T, p_slot, p_node, seed):
    """
    两段式入侵：每个 time slot 以 p_slot 概率发生攻击；
    若发生，则每个节点以 p_node 概率被入侵（返回 Node.ID 1-based 列表）。
    这样可以确保 noQ / Q 使用“同一套入侵序列”，对比公平。
    """
    rng = random.Random(int(seed))
    schedule = []
    for _ in range(int(T)):
        intruded = []
        if rng.random() < float(p_slot):
            for n in base_nodeSet:
                if rng.random() < float(p_node):
                    intruded.append(int(n.ID))  # Node.ID 1-based
        schedule.append(intruded)
    return schedule

def run_cic_timecost_once(cfg, base_nodeSet, schedule, use_qlearning: bool, quiet: bool = True):
    """
    返回：steps（跑了多少个 time slot）, time_ms_sum/mean/p95。
    """
    cfg.use_qlearning = bool(use_qlearning)
    patch_all_configs(cfg)

    # 每次对比都从相同初始网络开始
    ns = copy.deepcopy(base_nodeSet)

    times_ms = []
    status = 1

    # 可选：如果你在 CICnetwork 里实现了 reset_rl，建议在每次“Q 模式”开始前 reset
    # 注意：noQ 不需要 reset
    if use_qlearning and hasattr(CICnetwork, "reset_rl"):
        CICnetwork.reset_rl(cfg)

    # 计时：强烈建议 suppress stdout（print 会极大影响耗时）
    ctx = suppress_stdout(quiet)
    with ctx:
        for t in range(1, int(cfg.T) + 1):
            intruded_ids = schedule[t - 1]

            if bool(getattr(cfg, "enable_intrusion", False)) and intruded_ids:
                apply_intrusions_by_id(ns, intruded_ids)

            t0 = time.perf_counter()
            try:
                status, _ = CICnetwork.CICnetworkWithT(ns, intruded_ids)
            except Exception:
                status = 0
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            if status == 0 and bool(getattr(cfg, "stop_when_all_fail", True)):
                break

    arr = np.array(times_ms, dtype=float)
    return {
        "steps": int(len(times_ms)),
        "time_ms_sum": float(arr.sum()) if len(arr) else 0.0,
        "time_ms_mean": float(arr.mean()) if len(arr) else None,
        "time_ms_p95": float(np.percentile(arr, 95)) if len(arr) else None,
    }

def t_fail_to_lifetime(t_fail, T: int) -> int:
    """
    t_fail: 第一次失败时刻(1..T)，None=一直不fail
    lifetime: 成功持续的时间(0..T)，定义为：
      - None -> T
      - t_fail -> max(0, t_fail-1)
    """
    if t_fail is None:
        return int(T)
    try:
        tf = int(t_fail)
        return max(0, tf - 1)
    except Exception:
        return int(T)

def summarize_vals(vals):
    """
    输入：list[float]
    输出：mean/std/var/ci95_half/p10/cvar5/n
    """
    arr = np.array(vals, dtype=float)
    n = int(arr.size)
    if n == 0:
        return dict(mean=None, std=None, var=None, ci95_half=None, p10=None, cvar5=None, n=0)

    mean = float(arr.mean())
    if n > 1:
        std = float(arr.std(ddof=1))
        var = float(arr.var(ddof=1))
        ci95_half = float(1.96 * std / np.sqrt(n))
    else:
        std = 0.0
        var = 0.0
        ci95_half = 0.0

    p10 = float(np.percentile(arr, 10))
    p5 = float(np.percentile(arr, 5))
    cvar5 = float(arr[arr <= p5].mean())  # worst 5% average

    return dict(mean=mean, std=std, var=var, ci95_half=ci95_half, p10=p10, cvar5=cvar5, n=n)


# ---------------------------
# CSV
# ---------------------------
def csv_write(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] saved csv -> {out_path}")


# ---------------------------
# Main experiments
# ---------------------------
def main():
    cfg = load_config("config.yaml")
    patch_all_configs(cfg)

    exp_id = int(getattr(cfg, "experiment_id", 1))
    seeds = list(getattr(cfg, "seeds", [0]))

    quiet = bool(getattr(cfg, "quiet", False))
    if quiet:
        import sys, io
        sys.stdout = io.StringIO()

    rows = []
    agg_rows = []

    def build_one_network():
        # networkModel.networkModel() 返回 (nodeSet, clusterHeadSet, clusterSet)
        out = networkModel.networkModel()
        if isinstance(out, tuple):
            return out[0]
        return out

    # -------------------------
    # EXP 1: lifetime vs nodeNumber (no intrusion)
    # -------------------------
    if exp_id == 1:
        cfg.enable_intrusion = False

        # 1) 扫描的节点数量列表：优先读 cfg.sweep_nodeNumber，否则只跑 cfg.nodeNumber
        sweep_node_numbers = list(getattr(cfg, "sweep_nodeNumber", [int(cfg.nodeNumber)]))

        # 2) 我们关心的“生存时间”指标（用 t_fail_* 直接比较即可；越大越好）
        metrics = [
            "t_fail_MSPA", "t_fail_MaxFlow", "t_fail_EC2", "t_fail_CIC",
            "overall_stop_t",
            "energy_mean_MSPA", "energy_var_MSPA",
            "energy_mean_MaxFlow", "energy_var_MaxFlow",
            "energy_mean_EC2", "energy_var_EC2",
            "energy_mean_CIC", "energy_var_CIC",
        ]

        # None 处理：如果方法在 [1..T] 内从未 fail，则 t_fail=None
        none_as_Tplus1 = bool(getattr(cfg, "none_as_Tplus1", True))
        T = int(cfg.T)

        # 用于保存“每个 nodeNumber 的聚合结果”
        for nn in sweep_node_numbers:
            cfg.nodeNumber = int(nn)
            patch_all_configs(cfg)  # ★关键：把 nodeNumber patch 到各模块 config

            outs_this_nn = []  # 用于聚合(100个seed)
            for seed in seeds:
                set_seed(seed)
                # （可选）如果你担心 RL 跨 seed 污染，在每个 seed 前重置 CIC RL
                if bool(getattr(cfg, "reset_cic_each_run", False)):
                    if hasattr(CICnetwork, "reset_rl"):
                        # clear_q=True 表示每个seed独立训练；False 表示跨seed累积训练
                        CICnetwork.reset_rl(clear_q=True)

                nodeSet = build_one_network()
                base = copy.deepcopy(nodeSet)

                out = run_lifetime_simulation(cfg, base)
                out.update({
                    "experiment_id": 1,
                    "seed": int(seed),
                    "nodeNumber": int(nn),
                    "enable_intrusion": False,
                    "use_qlearning": bool(getattr(cfg, "use_qlearning", False)),
                    "use_path_qlearning": bool(getattr(cfg, "use_path_qlearning", False)),
                })

                rows.append(out)  # raw
                outs_this_nn.append(out)  # agg

            # --- 3) 对这个 nodeNumber 做聚合：mean/std/var ---
            agg = {
                "experiment_id": 1,
                "nodeNumber": int(nn),
                "n": len(outs_this_nn),
                "enable_intrusion": False,
                "use_qlearning": bool(getattr(cfg, "use_qlearning", False)),
                "use_path_qlearning": bool(getattr(cfg, "use_path_qlearning", False)),
            }

            for m in metrics:
                vals = []
                for o in outs_this_nn:
                    v = o.get(m, None)
                    if v is None and none_as_Tplus1:
                        v = T + 1
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass

                if len(vals) == 0:
                    agg[m + "_mean"] = None
                    agg[m + "_std"] = None
                    agg[m + "_var"] = None
                else:
                    arr = np.array(vals, dtype=float)
                    agg[m + "_mean"] = float(arr.mean())
                    agg[m + "_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                    agg[m + "_var"] = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0

            agg_rows.append(agg)

        if bool(getattr(cfg, "save_csv", True)):
            out_csv = str(getattr(cfg, "out_csv", "exp1_results.csv"))
            csv_write(agg_rows, out_csv)


    # -------------------------
    # EXP 2: static barrier counts no intrusion
    # -------------------------
    elif exp_id == 2:
        for seed in seeds:
            set_seed(seed)
            nodeSet = build_one_network()
            out = run_static_barrier_counts(cfg, nodeSet)
            out.update({"experiment_id": 2, "seed": seed, "nodeNumber": int(cfg.nodeNumber)})
            rows.append(out)

    # -------------------------
    # EXP 3: intrusion sensitivity sweep (p_slot, p_node) + CIC Q/noQ
    # -------------------------
    elif exp_id == 3:
        cfg.enable_intrusion = True
        sweep_p_slot = list(getattr(cfg, "sweep_p_slot", [float(getattr(cfg, "p_slot", 0.1))]))
        sweep_p_node = list(getattr(cfg, "sweep_p_node", [float(getattr(cfg, "p_node", 0.05))]))

        # key = (p_slot, p_node, use_q) -> list of out dicts
        group_outputs = defaultdict(list)


        for p_slot in sweep_p_slot:
            for p_node in sweep_p_node:
                # compare CIC noQ vs Q
                for use_q in [False, True]:
                    cfg.p_slot = float(p_slot)
                    cfg.p_node = float(p_node)
                    cfg.use_qlearning = bool(use_q)
                    cfg.use_path_qlearning = bool(use_q)
                    patch_all_configs(cfg)

                    # 跑100组实验
                    for seed in seeds:
                        set_seed(seed)

                        nodeSet = build_one_network()
                        base = copy.deepcopy(nodeSet)

                        out = run_lifetime_simulation(cfg, base)
                        out.update({
                            "experiment_id": 3,
                            "seed": seed,
                            "p_slot": float(p_slot),
                            "p_node": float(p_node),
                            "use_qlearning": bool(use_q),
                        })
                        # 1) 保存每次 seed 的原始结果
                        rows.append(out)
                        # 2) 同时加入到分组容器里，后面做均值
                        key = (float(p_slot), float(p_node), bool(use_q))
                        group_outputs[key].append(out)

        # ====== 统计每组的平均值/方差/标准差，并存起来 ======
        # ====== 统计每组的平均值/方差/标准差 + CI + 尾部 ======
        T = int(getattr(cfg, "T", 100))

        # 你原本统计的指标（保留）
        metrics = [
            "t_fail_MSPA", "t_fail_MaxFlow", "t_fail_EC2", "t_fail_CIC",
            "overall_stop_t",
        ]

        # 新增：基于寿命 lifetime 的鲁棒性统计（推荐你论文用这个）
        life_metrics = [
            ("MSPA", "t_fail_MSPA"),
            ("MaxFlow", "t_fail_MaxFlow"),
            ("EC2", "t_fail_EC2"),
            ("CIC", "t_fail_CIC"),
        ]

        for (p_slot, p_node, use_q), outs in group_outputs.items():
            agg = {
                "experiment_id": 3,
                "node_number": int(cfg.nodeNumber),
                "p_slot": float(p_slot),
                "p_node": float(p_node),
                "use_qlearning": bool(use_q),
                "n_seeds": len(outs),
                "T": T,
            }

            # -------- 1) 你原来的 t_fail / overall_stop_t mean/std/var（做一个小修复：None 不跳过，可按 T+1 或 T 处理）--------
            for m in metrics:
                vals = []
                for o in outs:
                    v = o.get(m, None)

                    # 对 t_fail_*：None -> T+1（表示到T都没fail，fail发生在T之后）
                    if v is None and m.startswith("t_fail_"):
                        v = T + 1

                    # 对 overall_stop_t：一般不会 None，若 None -> T
                    if v is None and m == "overall_stop_t":
                        v = T

                    try:
                        vals.append(float(v))
                    except Exception:
                        pass

                s = summarize_vals(vals)
                agg[m + "_mean"] = s["mean"]
                agg[m + "_std"] = s["std"]
                agg[m + "_var"] = s["var"]
                agg[m + "_ci95_half"] = s["ci95_half"]
                agg[m + "_p10"] = s["p10"]
                agg[m + "_cvar5"] = s["cvar5"]

            # -------- 2) 新增：寿命 life_* 的鲁棒性统计（mean/std/var/CI/尾部）--------
            for name, tf_key in life_metrics:
                life_vals = []
                success_cnt = 0  # 在T内未失败（t_fail=None）的次数

                for o in outs:
                    tf = o.get(tf_key, None)
                    if tf is None:
                        success_cnt += 1
                    life = t_fail_to_lifetime(tf, T)
                    life_vals.append(float(life))

                s = summarize_vals(life_vals)
                agg[f"life_{name}_mean"] = s["mean"]
                agg[f"life_{name}_std"] = s["std"]
                agg[f"life_{name}_var"] = s["var"]
                agg[f"life_{name}_ci95_half"] = s["ci95_half"]
                agg[f"life_{name}_p10"] = s["p10"]
                agg[f"life_{name}_cvar5"] = s["cvar5"]

                agg[f"life_{name}_success_rate"] = float(success_cnt) / float(len(outs)) if len(outs) > 0 else None

            agg_rows.append(agg)

        if bool(getattr(cfg, "save_csv", True)):
            out_csv = str(getattr(cfg, "out_csv", "exp3_results.csv"))
            csv_write(agg_rows, out_csv)


    # -------------------------
    # EXP 4: energyThreshold sensitivity (no intrusion) + CIC Q/noQ
    # -------------------------
    elif exp_id == 4:
        cfg.enable_intrusion = False
        sweep_th = list(getattr(cfg, "sweep_energyThreshold", [float(cfg.energyThreshold)]))

        for seed in seeds:
            set_seed(seed)
            for th in sweep_th:
                for use_q in [False, True]:
                    cfg.energyThreshold = float(th)
                    cfg.use_qlearning = bool(use_q)
                    patch_all_configs(cfg)

                    nodeSet = build_one_network()
                    base = copy.deepcopy(nodeSet)
                    out = run_lifetime_simulation(cfg, base)

                    out.update({
                        "experiment_id": 4, "seed": seed,
                        "energyThreshold": float(th),
                        "use_qlearning": bool(use_q),
                    })
                    rows.append(out)

    # -------------------------
    # EXP 5: energy consumption variance
    # (默认跑一次寿命实验/或你也可以打开 intrusion)
    # -------------------------
    elif exp_id == 5:
        # 你可以在 yaml 里决定是否 intrusion
        for seed in seeds:
            set_seed(seed)
            nodeSet = build_one_network()
            base = copy.deepcopy(nodeSet)
            out = run_lifetime_simulation(cfg, base)
            out.update({"experiment_id": 5, "seed": seed})
            rows.append(out)

    # -------------------------
    # EXP 6: ablation
    #   - enable_CIC
    #   - use_dynamic_weight_graph
    #   - use_qlearning
    # -------------------------
    elif exp_id == 6:
        cfg.enable_intrusion = False
        combos = list(itertools.product([False, True], [False, True], [False, True]))
        # (use_cic, dyn_weight, use_q)
        for seed in seeds:
            set_seed(seed)
            for use_cic, dyn_w, use_q in combos:
                cfg.enable_CIC = bool(use_cic)
                cfg.use_dynamic_weight_graph = bool(dyn_w)
                cfg.use_qlearning = bool(use_q)
                patch_all_configs(cfg)

                nodeSet = build_one_network()
                base = copy.deepcopy(nodeSet)
                out = run_lifetime_simulation(cfg, base)
                out.update({
                    "experiment_id": 6, "seed": seed,
                    "enable_CIC": bool(use_cic),
                    "use_dynamic_weight_graph": bool(dyn_w),
                    "use_qlearning": bool(use_q),
                })
                rows.append(out)

    # -------------------------
    # EXP 7: average power vs total energy
    # -------------------------
    elif exp_id == 7:
        cfg.enable_intrusion = bool(getattr(cfg, "enable_intrusion", False))  # 默认 false
        sweep_E = list(getattr(cfg, "sweep_total_energy", [cfg.nodeNumber * cfg.initialEnergy]))
        nodeN = int(getattr(cfg, "nodeNumber", 130))

        rows = []
        # 你如果做 RL vs noRL 对比，可以这里循环 use_q
        for use_q in [False, True]:
            cfg.use_qlearning = bool(use_q)
            patch_all_configs(cfg)

            for E_total in sweep_E:
                # 关键：用 E_total 控制 initialEnergy
                cfg.nodeNumber = nodeN
                cfg.initialEnergy = float(E_total) / float(nodeN)
                patch_all_configs(cfg)

                outs = []
                for seed in seeds:
                    set_seed(seed)
                    nodeSet = build_one_network()
                    base = copy.deepcopy(nodeSet)

                    out = run_lifetime_simulation(cfg, base)  # 需要它返回 avg_power_*
                    out.update({
                        "experiment_id": 7,
                        "seed": seed,
                        "total_energy": float(E_total),
                        "initialEnergy": float(cfg.initialEnergy),
                        "use_qlearning": bool(use_q),
                    })

                    rows.append(out)
                    outs.append(out)

                # ===== 聚合：对同一个 (E_total, use_qlearning) 做 100 seeds mean/std/var =====
                agg = {
                    "experiment_id": 7,
                    "total_energy": float(E_total),
                    "initialEnergy": float(cfg.initialEnergy),
                    "use_qlearning": bool(use_q),
                    "n_seeds": len(seeds),
                }

                metrics = [
                    "avg_power_MSPA", "avg_power_MaxFlow", "avg_power_EC2", "avg_power_CIC",
                    "t_run_MSPA", "t_run_MaxFlow", "t_run_EC2", "t_run_CIC",
                    "energy_cons_MSPA", "energy_cons_MaxFlow", "energy_cons_EC2", "energy_cons_CIC",
                ]

                for m in metrics:
                    vals = []
                    for o in outs:
                        v = o.get(m, None)
                        if v is None:
                            continue
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass

                    if len(vals) == 0:
                        agg[m + "_mean"] = None
                        agg[m + "_std"] = None
                        agg[m + "_var"] = None
                    else:
                        arr = np.array(vals, dtype=float)
                        agg[m + "_mean"] = float(arr.mean())
                        agg[m + "_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                        agg[m + "_var"] = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0

                agg_rows.append(agg)

        if bool(getattr(cfg, "save_csv", True)):
            out_csv = str(getattr(cfg, "out_csv", "exp7_results.csv"))
            csv_write(agg_rows, out_csv)

    # -------------------------
    # EXP 8: CIC time overhead (ms) of Q-learning vs noQ under intrusion rates
    # -------------------------
    elif exp_id == 8:
        # 1) 实验开关
        cfg.enable_MSPA = False
        cfg.enable_MaxFlow = False
        cfg.enable_EC2 = False
        cfg.enable_CIC = True

        cfg.enable_intrusion = True  # 本实验需要入侵
        cfg.stop_when_all_fail = True  # 路径断了就停止（否则后面也没意义）
        cfg.qtable_save_every = int(getattr(cfg, "qtable_save_every", 0))  # 建议设为0测纯开销

        # 2) sweep 参数（支持你 config.yaml 里写 sweep_p_slot / sweep_p_node）
        sweep_p_slot = list(getattr(cfg, "sweep_p_slot", [float(getattr(cfg, "p_slot", 0.1))]))
        sweep_p_node = list(getattr(cfg, "sweep_p_node", [float(getattr(cfg, "p_node", 0.05))]))

        raw_rows = []
        # 用于聚合：key=(p_slot,p_node) -> list[dict]
        bucket = defaultdict(list)

        for p_slot in sweep_p_slot:
            for p_node in sweep_p_node:
                cfg.p_slot = float(p_slot)
                cfg.p_node = float(p_node)

                for seed in seeds:
                    set_seed(seed)

                    # 构建网络：同一个 seed 下，noQ / Q 用同一张网络
                    nodeSet = build_one_network()
                    base = copy.deepcopy(nodeSet)

                    # 关键：入侵 schedule 预生成，确保 noQ / Q 完全一致
                    schedule = build_intrusion_schedule(base, cfg.T, cfg.p_slot, cfg.p_node, seed)

                    # -------- no Q-learning --------
                    out_noq = run_cic_timecost_once(cfg, base, schedule, use_qlearning=False, quiet=True)

                    # -------- Q-learning --------
                    out_q = run_cic_timecost_once(cfg, base, schedule, use_qlearning=True, quiet=True)

                    # 计算 overhead
                    overhead_ms = None
                    overhead_ratio = None
                    if out_noq["time_ms_mean"] is not None and out_q["time_ms_mean"] is not None:
                        overhead_ms = float(out_q["time_ms_mean"] - out_noq["time_ms_mean"])
                        if out_noq["time_ms_mean"] > 1e-12:
                            overhead_ratio = float(overhead_ms / out_noq["time_ms_mean"])

                    row = {
                        "experiment_id": 8,
                        "seed": int(seed),
                        "p_slot": float(cfg.p_slot),
                        "p_node": float(cfg.p_node),

                        "noq_steps": out_noq["steps"],
                        "noq_time_ms_sum": out_noq["time_ms_sum"],
                        "noq_time_ms_mean": out_noq["time_ms_mean"],
                        "noq_time_ms_p95": out_noq["time_ms_p95"],

                        "q_steps": out_q["steps"],
                        "q_time_ms_sum": out_q["time_ms_sum"],
                        "q_time_ms_mean": out_q["time_ms_mean"],
                        "q_time_ms_p95": out_q["time_ms_p95"],

                        "overhead_ms": overhead_ms,
                        "overhead_ratio": overhead_ratio,
                    }
                    raw_rows.append(row)
                    bucket[(float(cfg.p_slot), float(cfg.p_node))].append(row)

                print(f"[EXP8] done p_slot={p_slot} p_node={p_node} seeds={len(seeds)}")

        # 聚合（按入侵率点求 mean/std/var）
        agg_rows = []

        def _agg_stats(rows, key_name):
            vals = []
            for r in rows:
                v = r.get(key_name, None)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if len(vals) == 0:
                return None, None, None
            arr = np.array(vals, dtype=float)
            mean = float(arr.mean())
            var = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            return mean, std, var

        for (p_slot, p_node), rows_ in bucket.items():
            m1, s1, v1 = _agg_stats(rows_, "noq_time_ms_mean")
            m2, s2, v2 = _agg_stats(rows_, "q_time_ms_mean")
            mo, so, vo = _agg_stats(rows_, "overhead_ms")
            mr, sr, vr = _agg_stats(rows_, "overhead_ratio")

            agg_rows.append({
                "experiment_id": 8,
                "p_slot": float(p_slot),
                "p_node": float(p_node),
                "n": int(len(rows_)),

                "noq_time_ms_mean_mean": m1,
                "noq_time_ms_mean_std": s1,
                "noq_time_ms_mean_var": v1,

                "q_time_ms_mean_mean": m2,
                "q_time_ms_mean_std": s2,
                "q_time_ms_mean_var": v2,

                "overhead_ms_mean": mo,
                "overhead_ms_std": so,
                "overhead_ms_var": vo,

                "overhead_ratio_mean": mr,
                "overhead_ratio_std": sr,
                "overhead_ratio_var": vr,
            })

        if bool(getattr(cfg, "save_csv", True)):
            out_csv = str(getattr(cfg, "out_csv", "exp8_results.csv"))
            csv_write(agg_rows, out_csv)


    # -------------------------
    # EXP 9: average power vs intrusion rate (sweep p_slot, fixed p_node)
    # -------------------------
    elif exp_id == 9:
        # 开启入侵（two-stage）
        cfg.enable_intrusion = True

        sweep_p_slot = list(getattr(cfg, "sweep_p_slot", [float(getattr(cfg, "p_slot", 0.1))]))
        # p_node 固定：优先用 p_node_fixed，否则用 cfg.p_node
        p_node = float(getattr(cfg, "p_node", getattr(cfg, "p_node", 0.05)))

        rows = []  # raw per-seed
        agg_rows = []  # aggregated per (p_slot, use_q)

        # RL vs noRL 对比（你也可以只跑 True）
        for use_q in [False, True]:
            cfg.use_qlearning = bool(use_q)
            # 如果你还有 path-q 的开关，也同步
            if hasattr(cfg, "use_path_qlearning"):
                cfg.use_path_qlearning = bool(use_q)
            patch_all_configs(cfg)

            for p_slot in sweep_p_slot:
                cfg.p_slot = float(p_slot)
                cfg.p_node = float(p_node)  # 固定 p_node
                patch_all_configs(cfg)

                outs = []
                for seed in seeds:
                    set_seed(seed)

                    nodeSet = build_one_network()
                    base = copy.deepcopy(nodeSet)

                    out = run_lifetime_simulation(cfg, base)  # 需要它返回 avg_power_*
                    out.update({
                        "experiment_id": 11,
                        "seed": int(seed),
                        "p_slot": float(cfg.p_slot),
                        "p_node": float(cfg.p_node),
                        "use_qlearning": bool(use_q),
                    })

                    rows.append(out)
                    outs.append(out)

                # ===== 聚合：对同一个 (p_slot, use_qlearning) 做 mean/std/var =====
                agg = {
                    "experiment_id": 11,
                    "p_slot": float(cfg.p_slot),
                    "p_node": float(cfg.p_node),
                    "use_qlearning": bool(use_q),
                    "n_seeds": len(seeds),
                }

                metrics = [
                    "avg_power_MSPA", "avg_power_MaxFlow", "avg_power_EC2", "avg_power_CIC",
                    "t_run_MSPA", "t_run_MaxFlow", "t_run_EC2", "t_run_CIC",
                    "energy_cons_MSPA", "energy_cons_MaxFlow", "energy_cons_EC2", "energy_cons_CIC",
                    # 你如果还在 out 里返回寿命/失败时间，也可以加：
                    # "overall_stop_t", "t_fail_CIC", ...
                ]

                for m in metrics:
                    vals = []
                    for o in outs:
                        v = o.get(m, None)
                        if v is None:
                            continue
                        try:
                            vals.append(float(v))
                        except Exception:
                            pass

                    if len(vals) == 0:
                        agg[m + "_mean"] = None
                        agg[m + "_std"] = None
                        agg[m + "_var"] = None
                    else:
                        arr = np.array(vals, dtype=float)
                        agg[m + "_mean"] = float(arr.mean())
                        agg[m + "_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                        agg[m + "_var"] = float(arr.var(ddof=1)) if len(arr) > 1 else 0.0

                agg_rows.append(agg)

        # 保存：聚合表（必须）+ raw（可选）
        if bool(getattr(cfg, "save_csv", True)):
            out_csv = str(getattr(cfg, "out_csv", "exp9_results.csv"))
            csv_write(agg_rows, out_csv)

            # 可选保存 raw（逐 seed）
            out_raw = str(getattr(cfg, "out_csv_raw", "exp9_raw.csv"))
            csv_write(rows, out_raw)



    else:
        raise ValueError(f"Unknown experiment_id={exp_id}")

    # restore stdout if quiet
    if quiet:
        import sys
        content = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__
        print(content)

    # save / print
    if bool(getattr(cfg, "save_csv", True)):
        out_csv = str(getattr(cfg, "out_csv", "exp_results.csv"))
        csv_write(agg_rows, out_csv)

    # Also print summary
    for r in agg_rows[:10]:
        print(r)
    if len(agg_rows) > 10:
        print(f"... total rows={len(agg_rows)}")


if __name__ == "__main__":
    main()
