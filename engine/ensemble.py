"""Ensamblado ponderado, métricas secundarias y auto-evaluación (backtesting)."""
import math
import numpy as np
from . import models

DEFAULT_WEIGHTS = {"cold": 0.18, "recent_avoid": 0.10, "markov2": 0.15, "markov3": 0.15, "recency": 0.08, "cyclic": 0.12, "frecuencia": 0.08, "mlp": 0.14, "relative_freq": 0.10}


def validate(seq):
    seq = list(seq)
    if not 10 <= len(seq) <= 15:
        raise ValueError("La muestra debe tener entre 10 y 15 números")
    if any(isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 20 for n in seq):
        raise ValueError("Todos los números deben ser enteros entre 1 y 20")
    return seq


def shannon_entropy(seq) -> dict:
    counts = np.bincount(np.asarray(seq) - 1, minlength=20)
    p = counts[counts > 0] / len(seq)
    h = float(-(p * np.log2(p)).sum())
    mx = math.log2(20)
    return {"bits": round(h, 4), "max_bits": round(mx, 4), "normalizada": round(h / mx, 4)}


def _adaptive_weights(seq, entropy_normalized: float) -> dict:
    """Calcula pesos adaptativos basados en la entropía de la secuencia."""
    base = {
        "cold": 0.20,
        "recent_avoid": 0.12,
        "markov2": 0.12,
        "markov3": 0.12,
        "recency": 0.10,
        "cyclic": 0.12,
        "frecuencia": 0.12,
        "mlp": 0.10,
        "relative_freq": 0.10,
    }
    if entropy_normalized < 0.4:
        base["cyclic"] = 0.20
        base["markov3"] = 0.15
        base["mlp"] = 0.08
    elif entropy_normalized > 0.8:
        base["markov2"] = 0.18
        base["markov3"] = 0.15
        base["mlp"] = 0.14
    # Ajustar peso de relative_freq según la entropía
    if entropy_normalized < 0.3:
        base["relative_freq"] = 0.15
    elif entropy_normalized > 0.7:
        base["relative_freq"] = 0.08
    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


def _frequency_model(seq) -> np.ndarray:
    """Modelo de frecuencia: prioriza números menos frecuentes (números fríos)."""
    counts = np.zeros(20)
    for n in seq:
        counts[n - 1] += 1
    max_count = max(counts.max(), 1)
    freq = 1.0 / (counts + 0.1)
    return freq / freq.sum()


def component_probs(seq, weights=None) -> dict:
    return {
        "cold": models.cold_numbers(seq),
        "recent_avoid": models.recent_avoid(seq, window=5),
        "markov2": models.markov(seq, 2),
        "markov3": models.markov(seq, 3),
        "recency": models.recency_frequency(seq),
        "cyclic": models.cyclic_pattern(seq),
        "frecuencia": _frequency_model(seq),
        "mlp": models.mlp(seq),
        "relative_freq": models.relative_frequency(seq),
    }


def ensemble(seq, weights=None):
    weights = weights or DEFAULT_WEIGHTS
    comps = component_probs(seq)
    tw = sum(weights.values())
    p = sum(weights[k] * comps[k] for k in comps) / tw
    p = p / p.sum()
    entropy = shannon_entropy(seq)
    if np.std(p) > 0.08:
        std_factor = 1.0 - min(np.std(p) / 0.15, 0.5)
        p = 0.5 * p + 0.5 * std_factor * np.ones(20) / 20
        p = p / p.sum()
    return p, comps


def predict(seq, top_k: int = 5, weights=None) -> dict:
    seq = validate(seq)
    entropy_info = shannon_entropy(seq)
    weights = weights or _adaptive_weights(seq, entropy_info["normalizada"])
    p, comps = ensemble(seq, weights)
    order = np.argsort(-p, kind="stable")[:top_k]
    return {
        "muestra": seq,
        "predicciones": [
            {"numero": int(i + 1), "confianza_pct": round(float(p[i]) * 100, 2),
             "por_modelo_pct": {k: round(float(v[i]) * 100, 2) for k, v in comps.items()}}
            for i in order
        ],
        "par_impar": {"par_pct": round(float(p[1::2].sum()) * 100, 2),
                      "impar_pct": round(float(p[0::2].sum()) * 100, 2)},
        "rangos": {"1-10_pct": round(float(p[:10].sum()) * 100, 2),
                   "11-20_pct": round(float(p[10:].sum()) * 100, 2)},
        "entropia": entropy_info,
        "pesos": weights,
        "aviso": "Recomendación basada en priorizar números menos frecuentes / no salidos en la secuencia actual (evitando sobre-frecuentes como 17 o 19).",
    }


def backtest(history, window: int = 12, top_k: int = 5) -> dict:
    """Validación walk-forward: para cada t predice history[t] usando los `window`
    previos. Métricas: top-1, top-k, log-loss, Brier, comparadas con el azar."""
    history = list(history)
    hits1 = hitsk = n = 0
    ll = br = 0.0
    for t in range(window, len(history)):
        sample = history[t - window:t]
        entropy = shannon_entropy(sample)
        weights = _adaptive_weights(sample, entropy["normalizada"])
        p, _ = ensemble(sample, weights)
        y = history[t] - 1
        n += 1
        hits1 += int(np.argmax(p) == y)
        hitsk += int(y in np.argsort(-p, kind="stable")[:top_k])
        ll += -math.log(max(p[y], 1e-12))
        e = -p.copy(); e[y] += 1
        br += float((e ** 2).sum())
    if n == 0:
        raise ValueError("Historial insuficiente para backtesting")
    return {"evaluaciones": n, "top1": hits1 / n, f"top{top_k}": hitsk / n,
            "log_loss": ll / n, "brier": br / n,
            "baseline_azar": {"top1": 1 / 20, f"top{top_k}": top_k / 20, "log_loss": math.log(20)}}