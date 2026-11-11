# Unit 4: Hidden Markov Model & Viterbi Algorithm from Scratch

## 1. What the Algorithm Does

Physical HVAC systems do not switch operational states instantaneously at 1-minute intervals; thermal inertia and control hysteresis enforce temporal persistence.

Unit 4 models the time series as a **Discrete Hidden Markov Model (HMM)** and applies the **Viterbi dynamic programming algorithm** in log-probability space to recover the most likely latent state path:
$$\mathbf{q}^* = \arg\max_{\mathbf{q}} P(\mathbf{q}, \mathbf{o}_{1:T} | \lambda)$$

This filters out noisy, isolated sensor chattering and provides smoothed operational regime tracking.

## 2. Mathematical Foundation & Log-Space Trellis

Let:
- $\mathbf{S} = \{s_1, \dots, s_N\}$ be latent operating states.
- $\mathbf{V} = \{v_1, \dots, v_M\}$ be discrete observation symbols (derived from operating regime clusters).
- $\boldsymbol{\pi}$ be the initial state probability vector: $\pi_i = P(q_1 = s_i)$.
- $\mathbf{A}$ be the state transition probability matrix: $A_{ij} = P(q_{t+1} = s_j | q_t = s_i)$.
- $\mathbf{B}$ be the emission matrix: $B_{jk} = P(o_t = v_k | q_t = s_j)$.

### Log-Space Viterbi Recurrence
To avoid numerical underflow on multi-hundred step time series, calculations are evaluated in log-space:

1. **Initialization ($t = 1$):**
   $$V_1(i) = \ln \pi_i + \ln B_i(o_1), \quad \text{ptr}_1(i) = 0$$

2. **Induction ($t = 2, \dots, T$):**
   $$V_t(j) = \max_{1 \le i \le N} \left[ V_{t-1}(i) + \ln A_{ij} \right] + \ln B_j(o_t)$$
   $$\text{ptr}_t(j) = \arg\max_{1 \le i \le N} \left[ V_{t-1}(i) + \ln A_{ij} \right]$$

3. **Termination:**
   $$q_T^* = \arg\max_{1 \le i \le N} V_T(i), \quad \ln P^* = \max_{1 \le i \le N} V_T(i)$$

4. **Backtracking:**
   $$q_t^* = \text{ptr}_{t+1}(q_{t+1}^*), \quad \text{for } t = T-1, T-2, \dots, 1$$

## 3. Implementation Workflow

1. Ingest continuous telemetry and map to discrete observation symbols using trained K-Means operating regimes.
2. Formulate persistence transition prior ($A_{ii} = 0.95$) reflecting HVAC operating continuity.
3. Apply `viterbi_decode()` to the sequence.
4. Compare raw observation switching against smoothed Viterbi latent path.

## 4. Evaluated Results & Limitations

- Evaluated across 2,000 continuous test steps: successfully decoded optimal state trajectory with $\ln P^* = -2805.39$.
- **Crucial limitation:** HMM latent states represent statistical persistence regimes, not confirmed mechanical degradation stages. They cannot predict a future breakdown date without historical failure event logs.

