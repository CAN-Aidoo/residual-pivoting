"""Re-run the 40 in-distribution test episodes with trajectory logging (same seeds as evaluate.py).
Per episode: pre/post-motion step counts, first-motion step size and rate, angle after the
first-motion step relative to target, final error. Also checks reproducibility vs eval files."""
import json, numpy as np
from multiprocessing import Pool
from pivot_env import PivotEpisode, sample_object, LinearPolicy


def one(args):
    method, w, o, seed = args
    mode = method if method in ("model", "modelG") else method.split("_")[0]
    res = PivotEpisode(o, np.random.default_rng(seed)).run(mode, LinearPolicy(w) if w is not None else None, log=True)
    tr = res["traj"]; first = None
    for j, (k, F, rt, th, tc) in enumerate(tr):
        if th <= -0.03:
            prevF = tr[j - 1][1] if j > 0 else None
            first = dict(j=j, step=(1 - F / prevF) if prevF else None, rate=rt, th_true=tc, past=tc - o["target"])
            break
    return dict(method=method, i=o["i"], err=res["err"], held=res["held"], t=res["t_sim"], n=len(tr),
                pre=(first["j"] if first else len(tr)), first=first)


if __name__ == "__main__":
    g = np.random.default_rng(99); objs = [dict(sample_object(g), i=i) for i in range(40)]
    M = {"model": None, "modelG": None}
    for s in range(3):
        for fam in ("residual", "residualG", "pure"):
            M[f"{fam}_s{s}"] = np.load(f"w_{'pure' if fam == 'pure' else 'residual'}_s{s}.npy")
    jobs = [(m, w, o, 5000 + o["i"]) for m, w in M.items() for o in objs]
    with Pool(2) as p:
        rows = p.map(one, jobs)
    json.dump(rows, open("traj_indist.json", "w"), indent=1, default=float)
    ref = {(r["method"], r["i"]): r["err"] for r in json.load(open("eval_indist.json")) + json.load(open("eval_indist_guard.json"))}
    mism = [(r["method"], r["i"]) for r in rows if abs(ref[(r["method"], r["i"])] - r["err"]) > 1e-9]
    print("reproducibility mismatches:", len(mism), mism[:5])
