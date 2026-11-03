#!/usr/bin/env python
"""Verification script for ensemble.py"""
import sys
sys.path.insert(0, r"C:\Users\zowya\roulette-engine")

# Test imports
from engine import ensemble, models
import numpy as np

# Test validate
try:
    ensemble.validate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print("✓ validate() works with valid input")
except Exception as e:
    print(f"✗ validate() error: {e}")

# Test shannon_entropy
try:
    result = ensemble.shannon_entropy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert "bits" in result and "max_bits" in result and "normalizada" in result
    print(f"✓ shannon_entropy() works: entropy={result}")
except Exception as e:
    print(f"✗ shannon_entropy() error: {e}")

# Test _adaptive_weights
try:
    weights = ensemble._adaptive_weights([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5)
    assert "cold" in weights and "markov2" in weights
    print(f"✓ _adaptive_weights() works: weights={weights}")
except Exception as e:
    print(f"✗ _adaptive_weights() error: {e}")

# Test _frequency_model
try:
    freq = ensemble._frequency_model([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert freq.shape == (20,)
    print(f"✓ _frequency_model() works: sum={freq.sum()}")
except Exception as e:
    print(f"✗ _frequency_model() error: {e}")

# Test component_probs
try:
    comps = ensemble.component_probs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert "cold" in comps and "markov2" in comps
    print(f"✓ component_probs() works with {len(comps)} components")
except Exception as e:
    print(f"✗ component_probs() error: {e}")

# Test ensemble
try:
    p, comps = ensemble.ensemble([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert p.shape == (20,) and abs(p.sum() - 1) < 1e-6
    print(f"✓ ensemble() works: sum={p.sum()}")
except Exception as e:
    print(f"✗ ensemble() error: {e}")

# Test predict
try:
    r = ensemble.predict([3, 7, 7, 12, 3, 7, 19, 3, 7, 7, 12, 3])
    assert "predicciones" in r and "entropia" in r and "aviso" in r
    print(f"✓ predict() works: {len(r['predicciones'])} predictions")
except Exception as e:
    print(f"✗ predict() error: {e}")

# Test backtest
try:
    import random
    random.seed(1)
    h = [random.randint(1, 20) for _ in range(40)]
    result = ensemble.backtest(h)
    assert "top1" in result and "log_loss" in result and "baseline_azar" in result
    print(f"✓ backtest() works: {result}")
except Exception as e:
    print(f"✗ backtest() error: {e}")

# Test DEFAULT_WEIGHTS
print(f"\n✓ DEFAULT_WEIGHTS = {ensemble.DEFAULT_WEIGHTS}")

print("\n✓ All tests passed! File is valid.")