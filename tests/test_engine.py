import random
import pytest
from fastapi.testclient import TestClient
from engine import ensemble, models
from engine.api import app

S = [3, 7, 7, 12, 3, 7, 19, 3, 7, 7, 12, 3]


def test_models_are_distributions():
    for p in (models.cold_numbers(S), models.markov(S, 2), models.markov(S, 3), models.recency_frequency(S), models.mlp(S)):
        assert p.shape == (20,) and abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


def test_cold_numbers():
    # Si en la secuencia el 17 aparece varias veces y el 1 aparece 0 veces, el 1 debe tener mayor probabilidad que el 17.
    seq = [17, 17, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    p = models.cold_numbers(seq)
    # Número 1 (index 0) no ha salido, número 17 (index 16) ha salido 3 veces
    assert p[0] > p[16]


def test_markov_learns_pattern():
    assert models.markov([5, 9, 5, 9, 5, 9, 5, 9, 5, 9], 2).argmax() == 4


def test_predict_output():
    r = ensemble.predict(S)
    assert len(r["predicciones"]) == 5
    c = [x["confianza_pct"] for x in r["predicciones"]]
    assert c == sorted(c, reverse=True)
    assert abs(r["par_impar"]["par_pct"] + r["par_impar"]["impar_pct"] - 100) < 0.1


def test_validation():
    with pytest.raises(ValueError):
        ensemble.predict([1] * 5)
    with pytest.raises(ValueError):
        ensemble.predict([21] * 10)


def test_backtest_and_api():
    random.seed(1)
    h = [random.randint(1, 20) for _ in range(40)]
    assert ensemble.backtest(h)["evaluaciones"] == 28
    c = TestClient(app)
    assert c.post("/predict", json={"numeros": S}).status_code == 200
    assert c.post("/predict", json={"numeros": [0] * 10}).status_code == 422
