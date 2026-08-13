import numpy as np
from gbm_ai.training.finalize_classifier import align_models, paired_bootstrap_difference, choose_thresholds

def test_align_models():
    rows = {
        "a": [{"sample_id":"x","target":"0","probability_gbm":"0.1"},{"sample_id":"y","target":"1","probability_gbm":"0.9"}],
        "b": [{"sample_id":"y","target":"1","probability_gbm":"0.8"},{"sample_id":"x","target":"0","probability_gbm":"0.2"}],
    }
    ids, targets, probs = align_models(rows)
    assert ids == ["x", "y"]
    assert targets.tolist() == [0, 1]
    assert probs["a"].shape == (2,)

def test_paired_bootstrap():
    y=np.array([0,0,0,1,1,1,1,0])
    better=np.array([.05,.10,.20,.70,.80,.90,.95,.25])
    worse=np.array([.20,.45,.35,.55,.60,.65,.70,.40])
    r=paired_bootstrap_difference(y,better,worse,"roc_auc",200,42)
    assert r["delta_a_minus_b"] >= 0
    assert r["n_valid_bootstrap"] > 0

def test_three_band_thresholds():
    y=np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    p=np.array([.02,.05,.08,.10,.20,.30,.40,.60,.10,.25,.45,.55,.70,.80,.90,.95])
    r=choose_thresholds(y,p,.75,.75)
    assert r["T_low"] < r["T_high"]
    b=r["three_band_development_behavior"]
    assert b["gbm_not_suspected_band"]["total"] + b["indeterminate_band"]["total"] + b["gbm_suspected_band"]["total"] == len(y)
