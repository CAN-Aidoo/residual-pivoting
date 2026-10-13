"""Evaluate model / residual / pure policies on held-out objects.

Usage: python evaluate.py SET N   (SET in: indist, ood, current)
  indist : same distribution as training, fresh seed (99, as paper 1 Exp C)
  ood    : out-of-range objects (heavier, longer offset, wider friction)
  current: in-distribution objects, full motor-current probe + LS pipeline (paper 1)
"""
import sys, os, json, glob, numpy as np
from multiprocessing import Pool
from pivot_env import PivotEpisode, sample_object, score, LinearPolicy


def run_one(args):
    method, w, obj, seed, meas = args
    mode = method if method in ("model", "modelG") else method.split("_")[0]
    res = PivotEpisode(obj, np.random.default_rng(seed), meas=meas).run(mode, LinearPolicy(w) if w is not None else None)
    return dict(method=method, i=obj["i"], err=float(res["err"]), held=bool(res["held"]), t_sim=float(res["t_sim"]),
                probes=int(res["probes"]), score=float(score(res)), **{k: obj[k] for k in ("m", "d", "mu", "mut", "target")})


def main():
    which, n = sys.argv[1], int(sys.argv[2])
    g = np.random.default_rng({"ood": 77, "val": 123, "confirm": 2026}.get(which, 99))
    objs = [dict(sample_object(g, ood=(which == "ood")), i=i) for i in range(n)]
    meas = "current" if which == "current" else "surrogate"
    tag = os.environ.get("TAG", which)
    methods = {"model": None, "modelG": None}
    for f in sorted(glob.glob("w_*_s*.npy")):
        methods[f[2:-4]] = np.load(f)
        if f.startswith("w_residual"):
            methods["residualG" + f[10:-4]] = np.load(f)
            methods["residualR" + f[10:-4]] = np.load(f)
    only = os.environ.get("ONLY")
    if only: methods = {k: v for k, v in methods.items() if any(k.startswith(p) for p in only.split(","))}
    jobs = [(mname, w, o, 5000 + o["i"], meas) for mname, w in methods.items() for o in objs]
    with Pool(2) as p:
        rows = p.map(run_one, jobs)
    json.dump(rows, open(f"eval_{tag}.json", "w"), indent=1)
    for mname in methods:
        R = [r for r in rows if r["method"] == mname]
        e = np.array([r["err"] for r in R if r["held"]])
        print(f"{mname:14s} rms={np.sqrt(np.mean(e**2)):.4f} w05={np.mean(np.abs(e) <= .05):.2f} "
              f"drops={sum(not r['held'] for r in R)} t={np.mean([r['t_sim'] for r in R]) / 60:.2f}min "
              f"probes={np.mean([r['probes'] for r in R]):.1f} score={np.mean([r['score'] for r in R]):.3f}")


if __name__ == "__main__":
    main()
