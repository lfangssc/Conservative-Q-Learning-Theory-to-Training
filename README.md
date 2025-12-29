# Conservative-Q-Learning-Theory-to-Training

---

## Background: Why Conservative Q-Learning (CQL) Exists

In **offline reinforcement learning (Offline RL)**, the agent is given a **fixed dataset** of transitions:


$$\mathcal{D} = {(s_i, a_i, r_i, s'*i)}*{i=1}^{N}$$

collected by some unknown or suboptimal **behavior policy**:


$$\pi_\beta(a \mid s)$$

The agent **cannot interact with the environment** during training. It must learn purely from these logged samples.

---

### The Problem with Standard Q-Learning Offline

If we apply the standard Q-learning update in this setting:

$$Q(s,a) \leftarrow r + \gamma \max_{a'} Q(s', a')$$

we run into a serious issue:

* The `max` operator **extrapolates beyond the dataset**
* It assigns high Q-values to **unseen or rarely observed actions**
* These actions may be unsafe, infeasible, or catastrophic in reality
---

## Core Idea of CQL: Conservative (Pessimistic) Q-Values

**Conservative Q-Learning (CQL)** addresses this by explicitly discouraging high Q-values for **out-of-distribution (OOD)** actions.
Instead of letting Q-values freely grow for unseen actions, CQL **pushes them downward**, making the learned Q-function deliberately pessimistic.

---

## CQL Objective (Simplified)

CQL optimizes the following objective:

$$
\mathcal{L}_{\text{CQL}}(Q)=
\underbrace{\mathbb{E}_{(s,a,r,s')\sim \mathcal{D}}
\left[
\left(
Q(s,a) -
\left(
r + \gamma \, \mathbb{E}_{a'\sim \pi_Q(\cdot\mid s')}[Q(s',a')]
\right)
\right)^2
\right]}_{\text{Bellman error}} +
\alpha\ \underbrace{(\mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi(a \mid s)}[Q(s,a)] -
\mathbb{E}_{(s,a) \sim \mathcal{D}}[Q(s,a)])}_{\text{conservatism penalty}}
$$

---
---

## Key Components

* $\pi(a \mid s)$ 
  Action sampling distribution (e.g., uniform, current policy)

* $\pi_Q(a \mid s)$
  Policy induced by the current Q-function

* $\alpha  > 0$
  Conservatism coefficient

  * Larger → more pessimistic (safer, but possibly under-performing)
  * Smaller → closer to standard Q-learning (riskier)

---


# d3rlpy CQL training 

### Offline dataset

$$\mathcal{D}={(s_i,a_i,r_i,s'*i,d_i)}*{i=1}^{N}$$
Each training **step** = one update iteration (sample a minibatch from (\mathcal{D})).

---

### 1) Sample minibatch

$$(s,a,r,s',d)\sim \mathcal{D}$$

---

### 2) Critic target (SAC-style)

Sample next action:
$$a'\sim \pi_\phi(\cdot|s')$$

Target:
$$y=r+\gamma(1-d)\Big(\min_j Q_{\bar{\theta}*j}(s',a')-\alpha*{\text{temp}}\log\pi_\phi(a'|s')\Big)$$

---

### 3) Bellman (TD) loss

$$L_{\text{bellman}}=\mathbb{E}\big[(Q_\theta(s,a)-y)^2\big]$$

---

### 4) Conservative (CQL) loss (sample-based)

Sample actions at (s): policy actions + random/uniform actions
$$\mathcal{A}_{\text{samples}}(s)={a^{(\pi)}}\cup{a^{(\text{rand})}}$$

Penalty:
$$
L_{\text{cql}}
==============

\alpha_{\text{cql}},
\mathbb{E}\left[
\log\sum_{a\in \mathcal{A}*{\text{samples}}(s)}\exp(Q*\theta(s,a))
------------------------------------------------------------------

Q_\theta(s,a_{\text{data}})
-\tau
\right]
$$

---

### 5) Critic update

$$
L_{\text{critic}}=L_{\text{bellman}}+(\texttt{conservative_weight})\cdot L_{\text{cql}}
$$

---

### 6) Actor + temperature (SAC)

Actor (policy) update:
$$
L_{\text{actor}}=\mathbb{E}\left[\alpha_{\text{temp}}\log\pi_\phi(a|s)-\min_j Q_{\theta_j}(s,a)\right]
$$

Temperature update adjusts (\alpha_{\text{temp}}) toward a target entropy.

---

### What to log (to “see” CQL)

* TD loss: (L_{\text{bellman}})
* conservative loss: (L_{\text{cql}})
* (\alpha_{\text{cql}}) (if learned), (\alpha_{\text{temp}})
* mean (Q(s,a_{\text{data}})) vs mean (Q(s,a_{\pi}))


