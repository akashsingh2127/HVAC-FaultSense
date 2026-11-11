"""Unit tests for Discrete Hidden Markov Model and Viterbi Decoding."""

import numpy as np
import pytest

from hvac_faultsense.models.hmm_viterbi_scratch import DiscreteHMMViterbiScratch


def test_viterbi_deterministic_example():
    """Verify Viterbi decoding against a known textbook 2-state HMM problem.

    States: 0 (Sunny), 1 (Rainy)
    Observations: 0 (Walk), 1 (Shop), 2 (Clean)
    """
    hmm = DiscreteHMMViterbiScratch(n_states=2, n_observations=3)
    hmm.pi = np.array([0.6, 0.4])
    hmm.A = np.array([
        [0.7, 0.3], # Sunny -> Sunny: 0.7, Sunny -> Rainy: 0.3
        [0.4, 0.6], # Rainy -> Sunny: 0.4, Rainy -> Rainy: 0.6
    ])
    hmm.B = np.array([
        [0.6, 0.3, 0.1], # Sunny emissions: Walk=0.6, Shop=0.3, Clean=0.1
        [0.1, 0.4, 0.5], # Rainy emissions: Walk=0.1, Shop=0.4, Clean=0.5
    ])

    # Sequence of observations: Walk (0), Shop (1), Clean (2)
    obs = np.array([0, 1, 2])
    path, log_prob = hmm.viterbi_decode(obs)

    # Walk strongly suggests Sunny (0), Clean strongly suggests Rainy (1)
    assert len(path) == 3
    assert path[0] == 0 # Sunny
    assert path[2] == 1 # Rainy
    assert log_prob < 0.0 # Valid log probability


def test_viterbi_numerical_stability():
    """Verify log-space calculations prevent underflow on long sequences."""
    hmm = DiscreteHMMViterbiScratch(n_states=3, n_observations=3).initialize_uniform_or_random(random_state=42)

    # Long sequence of 1000 steps
    rng = np.random.default_rng(42)
    obs_long = rng.integers(0, 3, size=1000)

    path, log_prob = hmm.viterbi_decode(obs_long)
    assert len(path) == 1000
    assert not np.isnan(log_prob)
    assert not np.isinf(log_prob)
    assert set(path).issubset({0, 1, 2})

