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


# d3rlpy CQL Training Theory

This note summarizes the **theoretical training loop** behind **d3rlpy’s CQL (Conservative Q-Learning)** for continuous-control offline RL, and maps the core math terms to the “what happens each step” workflow.

---

## 1) Offline dataset

We start with a fixed offline dataset of transitions:

$$
\mathcal{D}=\left{(s_i,a_i,r_i,s'*i,d_i)\right}*{i=1}^{N}
$$

* (s_i): state / observation
* (a_i): action taken by the behavior policy (logged controller)
* (r_i): reward
* (s'_i): next state
* (d_i\in{0,1}): done flag (episode ended; used to stop bootstrapping)

**Training step semantics (like d3rlpy):**

* A “step” is a **parameter update iteration** (gradient updates), not “one transition consumed.”
* Each step samples a minibatch (\mathcal{B}\subset \mathcal{D}) (with replacement).

---

## 2) Per-step training loop (conceptual)

### Step 0 — Sample a minibatch

Sample a batch of transitions from the offline dataset:

$$
(s,a,r,s',d)\sim \mathcal{D}
$$

---

## 3) Critic target (SAC-style backup)

CQL in d3rlpy is built on a SAC-style backbone (actor + twin critics + target critics).

### Policy action at next state

Sample next action from the current policy:
$$
a' \sim \pi_\phi(\cdot\mid s')
$$

### Target value using target critics (twin + min)

Compute the bootstrapped target (soft backup includes entropy):

$$
y=

r+\gamma(1-d)\Big(
\min_{j\in{1,2}} Q_{\bar{\theta}_j}(s',a')

\alpha_{\text{temp}}\log\pi_\phi(a'\mid s')
\Big)
$$

* (Q_{\bar{\theta}_j}): **target critic networks** (slow-moving EMA copies)
* (\alpha_{\text{temp}}): **entropy temperature** (SAC), may be learned

---

## 4) Bellman (TD) loss

Each critic regresses toward the target:

$$
L_{\text{bellman}}(\theta)

\mathbb{E}*{(s,a,r,s',d)\sim\mathcal{D}}
\left[
\left(Q*\theta(s,a)-y\right)^2
\right]
$$

With **twin critics**, you compute this for (Q_{\theta_1}) and (Q_{\theta_2}).

---

## 5) Conservative (CQL) loss

The offline RL challenge: the learned policy may propose **out-of-distribution actions** that do not exist in the dataset; critics can overestimate those actions.

CQL adds a conservative penalty that pushes down Q-values for “broad” actions while anchoring Q-values on dataset actions.

### Sample actions at state (s)

At each (s) in the batch, sample a set of actions (\mathcal{A}_{\text{samples}}(s)), commonly including:

* (a\sim \pi_\phi(\cdot|s)) (policy actions)
* (a\sim \text{Unif}(\mathcal{A})) (random/uniform actions)

Let (\mathcal{A}_{\text{samples}}(s)) denote the union of these sampled sets.

### Sample-based log-sum-exp penalty

The “paper-style” conservative term (implemented via sampling) is:

$$
L_{\text{cql}}(\theta)

\alpha_{\text{cql}},
\mathbb{E}*{s\sim\mathcal{D}}
\left[
\log\sum*{a\in \mathcal{A}*{\text{samples}}(s)}
\exp\big(Q*\theta(s,a)\big)

 Q_\theta(s,a_{\text{data}})

\tau
\right]
$$

* (a_{\text{data}}): the dataset action paired with that state (from ((s,a)\in\mathcal{D}))
* (\alpha_{\text{cql}}): conservative multiplier (can be fixed or learned)
* (\tau): threshold used in the constrained/Lagrangian variant (“alpha_threshold” concept)

Intuition:

* (\log\sum\exp Q(s,a)) behaves like a “soft max” over many actions (including OOD candidates),
* subtracting (Q(s,a_{\text{data}})) makes dataset actions relatively preferred,
* (-\tau) sets how conservative we want to be.

---

## 6) Critic update

The critic is trained on **Bellman regression + conservative regularizer**:

$$
L_{\text{critic}}

L_{\text{bellman}}
+
(\texttt{conservative_weight})\cdot L_{\text{cql}}
$$

* `conservative_weight` is an explicit knob controlling the strength of the conservative penalty.

---

## 7) Actor update (SAC-style)

The actor is trained to choose actions that score high under the critic while maintaining entropy:

$$
L_{\text{actor}}(\phi)

\mathbb{E}*{s\sim\mathcal{D},,a\sim\pi*\phi(\cdot|s)}
\left[
\alpha_{\text{temp}}\log\pi_\phi(a\mid s)

\min_{j\in{1,2}}Q_{\theta_j}(s,a)
\right]
$$

This is the usual SAC “maximize Q + entropy” objective, using the **current critics**.

---

## 8) Temperature update (SAC entropy tuning)

If temperature is learned, adjust (\alpha_{\text{temp}}) to match a target entropy (H_{\text{target}}):

$$
L_{\text{temp}}(\alpha_{\text{temp}})

\mathbb{E}\left[
\alpha_{\text{temp}}\left(-\log\pi_\phi(a\mid s)-H_{\text{target}}\right)
\right]
$$

---

## 9) CQL alpha update (Lagrangian / constrained variant)

If using adaptive conservatism, update (\alpha_{\text{cql}}) so the conservative constraint around (\tau) is satisfied:

* (\alpha_{\text{cql}}) increases if the critic is not conservative enough
* decreases if it is overly conservative

(Conceptually: “enforce the conservative gap target defined by (\tau)”.)

---

## 10) Target critic update (Polyak / EMA)

Target networks track critics slowly:

$$
\bar{\theta}\leftarrow (1-\rho)\bar{\theta}+\rho\theta
$$

* (\rho) is the target sync coefficient (often called `tau` in codebases; don’t confuse with CQL’s threshold (\tau)).

---

## 11) What to log to “see the theory” during training

To connect the math to training behavior, monitor these curves:

* **Bellman / TD loss:** (L_{\text{bellman}})
* **Conservative loss magnitude:** (L_{\text{cql}})
* **CQL alpha (if learned):** (\alpha_{\text{cql}})
* **Temperature (if learned):** (\alpha_{\text{temp}})
* **Q-value drift checks:**

  * ( \mathbb{E}[Q(s,a_{\text{data}})] ) (on dataset actions)
  * ( \mathbb{E}[Q(s,a_{\pi})] ) (on policy actions)
  * optionally ( \mathbb{E}[Q(s,a_{\text{rand}})] ) (random/uniform actions)

A classic offline-RL failure mode is:

* (Q(s,a_\pi)) becomes much larger than (Q(s,a_{\text{data}})),
* the actor exploits this,
* real rollout return drops even while critic/actor losses look “reasonable.”

---

## 12) Summary: the bridge from config → theory

Your `CQLConfig(...)` hyperparameters in d3rlpy set learning rates and weights for exactly these sub-updates:

* Critic TD/Bellman regression
* Conservative penalty strength + sampling behavior
* Actor (policy) objective
* Temperature tuning
* (Optional) Lagrangian tuning of (\alpha_{\text{cql}})
* Target network EMA coefficient

That’s the concrete “dot line” between the paper formula and the d3rlpy API knobs.








