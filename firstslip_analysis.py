"""First-slip analysis from traj_indist.json -> firstslip_stats.json (all statistics quoted in Sec. 4.4)."""
import json, numpy as np
from scipy.stats import spearmanr, rankdata

R = json.load(open("traj_indist.json"))
T = {r["i"]: r["target"] for r in json.load(open("eval_indist.json")) if r["method"] == "model"}
fam_of = lambda m: "Mod" if m in ("model", "modelG") else {"residual": "Res", "residualG": "ResG", "pure": "Pur"}[m.split("_")[0]]
X = [r for r in R if r["first"] and r["first"]["step"] is not None]
rot = np.array([-r["first"]["th_true"] for r in X]); step = np.array([r["first"]["step"] for r in X])
rate = np.array([r["first"]["rate"] for r in X]); tgt = np.array([-T[r["i"]] for r in X])
qs = np.arccos(1 - step); fam = np.array([fam_of(r["method"]) for r in X])


def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(ry), res(rx))[0, 1])


out = dict(n_episodes=len(X), n_objects=len(set(r["i"] for r in X)), rot_med=float(np.median(rot)),
           rho_step=float(spearmanr(rot, step)[0]), rho_rate=float(spearmanr(rot, rate)[0]), rho_step_rate=float(spearmanr(step, rate)[0]),
           prho_step=partial_spearman(rot, step, rate), prho_rate=partial_spearman(rot, rate, step),
           exceed=float(np.mean(rot > qs)), exceed_by_fam={f: float(np.mean((rot > qs)[fam == f])) for f in ("Mod", "Res", "ResG", "Pur")},
           rot_med_by_fam={f: float(np.median(rot[fam == f])) for f in ("Mod", "Res", "ResG", "Pur")})
of = np.array([r["first"]["past"] < -0.05 for r in X])
sh = (tgt >= 0.3) & (tgt < 0.5)
out.update(ov_shallow=float(of[sh].mean()), ov_shallow_n_obj=len(set(np.array([r["i"] for r in X])[sh])),
           ov_mid=float(of[(tgt >= 0.5) & (tgt < 0.8)].mean()), ov_deep=float(of[tgt >= 0.8].mean()))
json.dump(out, open("firstslip_stats.json", "w"), indent=1); print(json.dumps(out, indent=1))
