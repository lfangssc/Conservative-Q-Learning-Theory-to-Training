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

```md
## d3rlpy CQL training (theory → implementation map)

### Offline dataset

We assume an offline dataset of transitions:

D = { (s_i, a_i, r_i, s'_i, done_i) } for i = 1..N

Each training *step* = one update iteration:
- sample a minibatch from D
- run critic / actor / (optional) temperature + CQL-alpha updates

---

### 1) Sample minibatch

(s, a, r, s', done) ~ D

---

### 2) Critic target (SAC-style backup)

Sample next action from current policy:

a' ~ pi_phi(. | s')

Compute target value using target critics (twin critics + min) and entropy term:

y = r + gamma * (1 - done) * ( min_j Q_target_j(s', a') - alpha_temp * log pi_phi(a' | s') )

---

### 3) Bellman (TD) loss

L_bellman = E[ ( Q_theta(s, a) - y )^2 ]

(with twin critics, compute this for Q1 and Q2)

---

### 4) Conservative (CQL) loss (sample-based)

At each state s, sample a set of actions (controlled by n_action_samples), usually:
- a^(pi)   sampled from current policy pi_phi(.|s)
- a^(rand) sampled from uniform/random actions

A_samples(s) = { a^(pi) } U { a^(rand) }

Conservative penalty (sample-based approximation):

L_cql = alpha_cql * E[
  log sum_{a in A_samples(s)} exp( Q_theta(s, a) )
  - Q_theta(s, a_data)
  - tau
]

where a_data is the dataset action paired with s in the minibatch.

---

### 5) Critic update

L_critic = L_bellman + (conservative_weight) * L_cql

---

### 6) Actor + temperature (SAC)

Actor (policy) update:

L_actor = E[ alpha_temp * log pi_phi(a | s) - min_j Q_theta_j(s, a) ]

Temperature update:
- adjusts alpha_temp toward a target entropy (prevents policy collapse)

(Optional) CQL alpha update (Lagrangian):
- adjusts alpha_cql so the conservative constraint around threshold tau is satisfied
  (alpha_threshold is used for this behavior)

---

### What to log (to “see” CQL inside training)

- TD / Bellman loss: L_bellman
- Conservative loss:  L_cql
- alpha_cql (if learned) and alpha_temp
- Q-value scale checks:
  - mean Q(s, a_data)  (dataset actions)
  - mean Q(s, a_pi)    (policy actions)
  - optionally mean Q(s, a_rand) (random/uniform actions)
```
