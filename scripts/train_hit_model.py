"""Train the hit model and compare ranking quality on held-out missions (contracts/hit_model.md).

    python scripts/train_hit_model.py   -> artifacts/hit_model.json, artifacts/hit_model_eval.json
"""
import json
import os
import pickle
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import scenarios  # noqa: E402
from core.hit_model import HIT_FEATURES, HitModel, assert_train_seeds  # noqa: E402

MODEL_PATH = "artifacts/hit_model.json"

D = "artifacts/hit_data"


def load(split):
    z = np.load(f"{D}/{split}.npz")
    return {k: z[k] for k in z.files}


def ranking(y, w, s):
    return dict(auc_pr=float(average_precision_score(y, s, sample_weight=w)),
                roc_auc=float(roc_auc_score(y, s, sample_weight=w)))


def main():
    eval_only = "--eval-only" in sys.argv
    tr, va, ev = (None, None, load("eval")) if eval_only else (load("train"), load("val"), load("eval"))
    if eval_only:
        prev = json.load(open("artifacts/hit_model_eval.json"))
        params, best_it, val_best = prev["params"], prev["best_iteration"], prev["val_auc_pr_best"]
    else:
        params, best_it, val_best = train(tr, va)
    evaluate(ev, params, best_it, val_best)


def train(tr, va):
    assert_train_seeds(np.unique(tr["seed"]))
    assert_train_seeds(np.unique(va["seed"]))

    dtr = xgb.DMatrix(tr["X"], label=tr["y"], weight=tr["w"], feature_names=HIT_FEATURES)
    dva = xgb.DMatrix(va["X"], label=va["y"], weight=va["w"], feature_names=HIT_FEATURES)
    params = dict(objective="binary:logistic", eval_metric="aucpr", tree_method="hist", max_depth=6, eta=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=0, nthread=os.cpu_count())
    hist = {}
    booster = xgb.train(params, dtr, num_boost_round=1500, evals=[(dva, "val")], early_stopping_rounds=60,
                        evals_result=hist, verbose_eval=100)
    booster.save_model(MODEL_PATH)
    return params, int(booster.best_iteration), float(max(hist["val"]["aucpr"]))


def evaluate(ev, params, best_it, val_best):
    # Score through the exact path the scheduler uses (all trees in the saved file); see BUGLOG 2026-10-01.
    model = HitModel.load(MODEL_PATH)
    booster = model.booster
    assert np.unique(ev["seed"]).min() >= 1000, "evaluation data must come from held-out seeds"
    y, w, X = ev["y"], ev["w"], ev["X"]
    ours = model.score(X)
    old = pickle.load(open("smart_scan_xgboost.pkl", "rb"))  # inspected: only xgboost/numpy/builtins classes
    old_p = old.predict_proba(ev["legacy"].astype(np.float64))[:, 1]
    f = HIT_FEATURES.index
    rows = {
        "hit_model (ours)": ranking(y, w, ours),
        "smart_scan_xgboost.pkl (supplied)": ranking(y, w, old_p),
        "novel_mean alone": ranking(y, w, X[:, f("novel_mean")]),
        "window alone": ranking(y, w, X[:, f("window")]),
        "staleness alone": ranking(y, w, X[:, f("staleness")]),
    }
    per_suite = {}
    for i, s in enumerate(scenarios.SUITES):
        m = ev["suite"] == i
        per_suite[s] = {"hit_model": ranking(y[m], w[m], ours[m]), "supplied": ranking(y[m], w[m], old_p[m]),
                        "positive_rate": float(y[m].mean())}
    out = dict(no_skill_auc_pr=float((w * y).sum() / w.sum()), eval_rows=int(len(y)), eval_positives=int(y.sum()),
               best_iteration=best_it, val_auc_pr_best=val_best,
               n_trees_scored=int(booster.num_boosted_rounds()), scored_with="HitModel.load(artifacts/hit_model.json).score",
               overall=rows, per_suite=per_suite, params=params,
               importance_gain={k: round(v, 2) for k, v in sorted(booster.get_score(importance_type="gain").items(),
                                                                   key=lambda kv: -kv[1])})
    json.dump(out, open("artifacts/hit_model_eval.json", "w"), indent=1)
    print(f"no-skill AUC-PR {out['no_skill_auc_pr']:.4f}   trees scored {out['n_trees_scored']}")
    for k, v in rows.items():
        print(f"  {k:36s} AUC-PR {v['auc_pr']:.4f}   ROC-AUC {v['roc_auc']:.4f}")
    for s, v in per_suite.items():
        print(f"  {s}: ours {v['hit_model']['auc_pr']:.4f}  supplied {v['supplied']['auc_pr']:.4f}  base {v['positive_rate']:.4f}")
    print("gain:", list(out["importance_gain"].items())[:8])


if __name__ == "__main__":
    main()
