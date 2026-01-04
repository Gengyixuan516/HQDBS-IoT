# HQDBS-IoT: Hierarchical Q-Learning for Dynamic Barrier Scheduling & Recovery in IoT Networks

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#requirements)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

> **Toward Effective, Efficient, and Trustworthy Hierarchical Q-Learning Agents for Dynamic Barrier Scheduling and Recovery in IoT Networks**  
> This repository provides a reference implementation of **HQDBS** (Hierarchical Q-learning Dynamic Barrier Scheduling), a lightweight hierarchical RL framework for **continuous barrier scheduling + recovery** under energy depletion and node intrusion/failures.

---

## ✨ What is HQDBS?

Barrier coverage is essential for security scenarios such as perimeter protection and intrusion detection. In real IoT deployments, however, **energy decay**, **random failures**, and **adversarial intrusions** continuously change network feasibility, making static barrier construction insufficient.

**HQDBS** treats barrier maintenance as a **continuous online decision process** rather than static rules:
- A **global agent (Path-Q)** performs **repair-aware path planning**.
- Distributed **local agents (RA-Q)** conduct **node-group scheduling** to close coverage gaps under resource constraints.
- A **plan-then-commit** protocol verifies feasibility *before* activating nodes, and **structured replanning** isolates local failures to improve trustworthiness.

---

## 🚀 Key Contributions / Highlights

- **Framework:** HQDBS combines real-time **graph analytics** with lightweight **tabular Q-learning** for adaptive decision-making in dynamic IoT environments.
- **Hierarchical learning:** Unifies **path scheduling** and **path recovery** via cooperative **Path-Q + RA-Q** agents (avoids heavy deep models and supports low-overhead online operation).
- **Trustworthy execution:** A practical online mechanism with **plan-then-commit** and **structured repair/replanning**.
- **Empirical gains:** Improves barrier constructibility, extends network lifetime, and enhances robustness under intrusion while maintaining low latency and moderate overhead.

---

## 🧠 Method Overview (High-level)

**Core components**
1. **Dynamic point-weighted graph** as the real-time analytical backbone (energy, redundancy, regional availability).
2. **Path-Q (global):** chooses which barrier path to run next (repair-aware planning).
3. **RA-Q (local):** schedules node groups inside reconstruction areas (fine-grained allocation).
4. **Plan-then-commit:** first generate a full plan in memory; only commit node status updates after feasibility checks.
5. **Structured recovery:** if any RA fails, mark region blocked and trigger Path-Q replanning to avoid cascading failures.

<p align="center">
  <img src="pic\image.png" width="720">
</p>

---

## 📊 Experimental Results (from the paper)

Below are example results you can showcase in the README (place regenerated figures under `assets/` or `docs/figures/`).

### 1) Static barrier construction (baseline sanity check)
- **Barrier paths vs. number of sensing nodes:** HQDBS constructs **more barrier paths** than MSPA / MaxFlow as node count increases (static setting).
- **Path quantity vs. network scale:** HQDBS remains more scalable as the sensing area grows (see Table II in the paper).


<p align="center">
  <img src="pic\image-1.png" width="320">
  <img src="pic\image-2.png" width="320">
</p>


### 2) Dynamic conditions (no intrusion)
- **Network lifetime vs. node count:** HQDBS scales better with density.  
  Example (paper): when nodes increase **50 → 150**, HQDBS lifetime grows roughly **~40 → ~166**, while MSPA/MaxFlow rise only to about **~110 / ~116**.
- **Energy efficiency vs. total energy:** energy efficiency is defined as **total energy consumption / total operating time**; HQDBS consistently achieves **lower** (better) values than baselines.


<p align="center">
  <img src="pic\image-3.png" width="320">
  <img src="pic\image-4.png" width="320">
</p>


### 3) Node intrusion / failures
- **Intrusion probability vs lifetime:** lifetime decreases as intrusion increases for all methods, but HQDBS stays substantially higher across the intrusion space.
- **Energy efficiency vs p_slot (with p_node fixed):** HQDBS maintains lower average power expenditure under intrusion.


<p align="center">
  <img src="pic\image-5.png" width="320">
</p>

<p align="center">
  <img src="pic\image-6.png" width="600">
</p>

### 4) Scheduling Overhead under Intrusion

To evaluate the runtime cost introduced by the AI agent, we reconstruct a **no Q-learning** version using mature heuristic scheduling and compare it with HQDBS. We report three complementary metrics under varying slot-level intrusion probability `p_slot` and per-node intrusion probability `p_node`:

- **Decision-making time (ms):** increases moderately with intrusion severity due to more frequent replanning/repair, but remains **below ~8 ms** even under the most severe setting.
- **Absolute overhead (ms):** remains consistently low across intrusion levels.
- **Overhead ratio (%):** grows gradually as intrusion becomes more severe (especially at higher `p_slot`), yet stays within a moderate range, indicating a favorable robustness–efficiency trade-off.

<p align="center">
  <img src="pic\image-7.png" width="720">
</p>

---


