"""Hidden Markov Model (HMM) and Viterbi Decoding Implemented from Scratch.

Uses NumPy and log-space dynamic programming to prevent numerical underflow,
recovering the most likely latent operating state sequence over time.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


class DiscreteHMMViterbiScratch:
    """Discrete Hidden Markov Model with log-space Viterbi path decoding."""

    def __init__(
        self,
        n_states: int = 4,
        n_observations: int = 4,
        pseudocount: float = 1e-6,
    ):
        self.n_states = n_states
        self.n_observations = n_observations
        self.pseudocount = pseudocount

        # Parameters:
        # pi: initial state distribution, shape (n_states,)
        # A: transition matrix, A[i, j] = P(s_t+1 = j | s_t = i), shape (n_states, n_states)
        # B: emission matrix, B[j, k] = P(o_t = k | s_t = j), shape (n_states, n_observations)
        self.pi: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None

    def initialize_uniform_or_random(self, random_state: Optional[int] = 42) -> "DiscreteHMMViterbiScratch":
        """Initialize parameters with high self-transition probability (persistence)."""
        rng = np.random.default_rng(random_state)

        # Initial probabilities
        pi = np.full(self.n_states, 1.0 / self.n_states)

        # Transition matrix with diagonal persistence (HVAC regimes persist over minutes)
        A = np.full((self.n_states, self.n_states), 0.05 / (self.n_states - 1))
        for i in range(self.n_states):
            A[i, i] = 0.95
        # Re-normalize rows
        A = A / np.sum(A, axis=1, keepdims=True)

        # Emission matrix
        B = np.full((self.n_states, self.n_observations), 1.0 / self.n_observations)
        # Slight diagonal bias
        for i in range(min(self.n_states, self.n_observations)):
            B[i, i] += 0.5
        B = B / np.sum(B, axis=1, keepdims=True)

        self.pi = pi
        self.A = A
        self.B = B
        return self

    def fit_supervised(
        self,
        observation_seq: np.ndarray,
        state_seq: np.ndarray,
    ) -> "DiscreteHMMViterbiScratch":
        """Fit HMM parameters from known/aligned state and observation sequences."""
        obs = np.asarray(observation_seq, dtype=int)
        states = np.asarray(state_seq, dtype=int)
        T = len(obs)

        # 1. Estimate initial state probability
        pi_counts = np.zeros(self.n_states) + self.pseudocount
        pi_counts[states[0]] += 1.0
        self.pi = pi_counts / np.sum(pi_counts)

        # 2. Estimate transition matrix A
        A_counts = np.zeros((self.n_states, self.n_states)) + self.pseudocount
        for t in range(T - 1):
            A_counts[states[t], states[t + 1]] += 1.0
        self.A = A_counts / np.sum(A_counts, axis=1, keepdims=True)

        # 3. Estimate emission matrix B
        B_counts = np.zeros((self.n_states, self.n_observations)) + self.pseudocount
        for t in range(T):
            B_counts[states[t], obs[t]] += 1.0
        self.B = B_counts / np.sum(B_counts, axis=1, keepdims=True)

        return self

    def viterbi_decode(self, observation_seq: np.ndarray) -> Tuple[np.ndarray, float]:
        """Decode the most likely latent state sequence using log-space dynamic programming.

        Recurrence:
            V_1(i) = ln(pi_i) + ln(B_i(o_1))
            V_t(j) = max_i [ V_t-1(i) + ln(A_ij) ] + ln(B_j(o_t))
            ptr_t(j) = argmax_i [ V_t-1(i) + ln(A_ij) ]

        Returns:
            best_path: Array of state indices of shape (T,).
            log_probability: The log likelihood of the optimal sequence.
        """
        if self.pi is None or self.A is None or self.B is None:
            raise ValueError("HMM parameters are not set. Fit or initialize first.")

        obs = np.asarray(observation_seq, dtype=int)
        T = len(obs)
        if T == 0:
            return np.array([], dtype=int), 0.0

        # Precompute log probabilities with eps to prevent log(0)
        eps = 1e-15
        log_pi = np.log(np.maximum(self.pi, eps))
        log_A = np.log(np.maximum(self.A, eps))
        log_B = np.log(np.maximum(self.B, eps))

        # Dynamic programming trellis
        # V[t, j] stores max log prob ending at state j at time t
        V = np.zeros((T, self.n_states))
        backpointers = np.zeros((T, self.n_states), dtype=int)

        # Initialization at t = 0
        first_obs = obs[0]
        V[0, :] = log_pi + log_B[:, first_obs]

        # Forward trellis induction
        for t in range(1, T):
            curr_obs = obs[t]
            for j in range(self.n_states):
                # Candidate scores from all previous states i
                scores = V[t - 1, :] + log_A[:, j]
                best_prev = int(np.argmax(scores))
                backpointers[t, j] = best_prev
                V[t, j] = scores[best_prev] + log_B[j, curr_obs]

        # Termination: find best final state
        best_last_state = int(np.argmax(V[T - 1, :]))
        best_log_prob = float(V[T - 1, best_last_state])

        # Backtracking path reconstruction
        best_path = np.zeros(T, dtype=int)
        best_path[T - 1] = best_last_state
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointers[t + 1, best_path[t + 1]]

        return best_path, best_log_prob

