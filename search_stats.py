"""Search-phase behaviour and error-sign counts used in the paper text."""
import json, numpy as np
from pivot_env import PivotEpisode, LinearPolicy, sample_object
R = json.load(open("eval_indist.json")) + json.load(open("eval_indist_guard.json"))
by = {}
for r in R: by.setdefault(r["method"], {})[r["i"]] = r
out = {"under": {m: int(sum(v[i]["held"] and v[i]["err"] > 0.05 for i in v)) for m, v in by.items()},
       "obj7": dict(model=by["model"][7]["err"], modelG=by["modelG"][7]["err"], target=by["model"][7]["target"])}
g = np.random.default_rng(99); objs = [sample_object(g) for _ in range(10)]
for name, mode, w in [("model", "model", None)] + [(f"residual_s{s}", "residual", np.load(f"w_residual_s{s}.npy")) for s in range(3)]:
    ratios, rates = [], []
    for j, o in enumerate(objs):
        tr = PivotEpisode(o, np.random.default_rng(5000 + j)).run(mode, LinearPolicy(w) if w is not None else None, log=True)["traj"]
        for a, b in zip(tr[:-1], tr[1:]):
            if a[3] > -0.03: ratios.append(b[1] / a[1]); rates.append(b[2])
    out[name] = dict(step=float(1 - np.mean(ratios)), rate=float(np.mean(rates)))
json.dump(out, open("search_stats.json", "w"), indent=1); print(out)
