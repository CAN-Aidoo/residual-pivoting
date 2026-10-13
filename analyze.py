"""Aggregate paper-2 results: per-seed metrics, seed mean/sd, paired tests vs the model controller."""
import json, numpy as np
from scipy.stats import wilcoxon

SETS = {"indist": ["eval_indist.json", "eval_indist_guard.json"], "ood": ["eval_ood_a.json", "eval_ood_b.json"],
        "current": ["eval_current_a.json", "eval_current_b.json"], "val": ["eval_val.json", "eval_val_rate.json"], "confirm": ["eval_confirm.json"]}
FAM = {"model": ["model"], "modelG": ["modelG"], "pure": ["pure_s0", "pure_s1", "pure_s2"],
       "residual": ["residual_s0", "residual_s1", "residual_s2"], "residualG": ["residualG_s0", "residualG_s1", "residualG_s2"],
       "residualR": ["residualR_s0", "residualR_s1", "residualR_s2"]}


def metrics(R):
    e = np.array([r["err"] for r in R if r["held"]])
    return dict(n=len(R), rms=float(np.sqrt(np.mean(e ** 2))), med=float(np.median(np.abs(e))),
                w05=float(np.mean([r["held"] and abs(r["err"]) <= 0.05 for r in R])),
                over=int(sum(r["held"] and r["err"] < -0.05 for r in R)), drops=int(sum(not r["held"] for r in R)),
                t_min=float(np.mean([r["t_sim"] for r in R]) / 60), probes=float(np.mean([r["probes"] for r in R])),
                score=float(np.mean([r["score"] for r in R])))


out = {}
for sname, files in SETS.items():
    rows = sum([json.load(open(f)) for f in files], [])
    by = {}
    for r in rows: by.setdefault(r["method"], []).append(r)
    for v in by.values(): v.sort(key=lambda r: r["i"])
    S = out[sname] = {}
    for fam, members in FAM.items():
        per = {m: metrics(by[m]) for m in members if m in by}
        if not per: continue
        agg = {k: (float(np.mean([p[k] for p in per.values()])), float(np.std([p[k] for p in per.values()], ddof=1)) if len(per) > 1 else 0.0)
               for k in ("rms", "med", "w05", "over", "drops", "t_min", "probes", "score")}
        S[fam] = dict(per_seed=per, mean_sd=agg)
        # paired Wilcoxon vs model on time and |error| (pooled over seeds: per-object seed-mean)
        if fam not in ("model",) and "model" in by and all(m in by for m in members):
            ref = by["model"]
            dt = np.mean([[r["t_sim"] - q["t_sim"] for r, q in zip(by[m], ref)] for m in members], 0)
            de = np.mean([[abs(r["err"]) - abs(q["err"]) for r, q in zip(by[m], ref)] for m in members], 0)
            S[fam]["vs_model"] = dict(dt_min_mean=float(dt.mean() / 60), p_time=float(wilcoxon(dt).pvalue) if np.any(dt) else 1.0,
                                     dabs_err_mean=float(de.mean()), p_err=float(wilcoxon(de).pvalue) if np.any(de) else 1.0)
    # extra paired tests: guarded residual vs model+guard; pure vs residual
    def paired(fa, fb):
        A, Bm = FAM[fa], FAM[fb]
        if not all(m in by for m in A + Bm): return None
        ea = np.mean([[abs(r["err"]) for r in by[m]] for m in A], 0); eb = np.mean([[abs(r["err"]) for r in by[m]] for m in Bm], 0)
        ta = np.mean([[r["t_sim"] for r in by[m]] for m in A], 0); tb = np.mean([[r["t_sim"] for r in by[m]] for m in Bm], 0)
        return dict(dabs_err=float((ea - eb).mean()), p_err=float(wilcoxon(ea - eb).pvalue), dt_min=float((ta - tb).mean() / 60), p_time=float(wilcoxon(ta - tb).pvalue))
    S["pairs"] = {"residualG_vs_modelG": paired("residualG", "modelG"), "pure_vs_residual": paired("pure", "residual")}
json.dump(out, open("results_summary.json", "w"), indent=1)
for sname, S in out.items():
    print("==", sname)
    print("  pairs", S.get("pairs"))
    for fam, d in S.items():
        if fam == "pairs": continue
        a = d["mean_sd"]; vs = d.get("vs_model", {})
        print(f"  {fam:10s} rms={a['rms'][0]:.3f}±{a['rms'][1]:.3f} w05={a['w05'][0]:.2f}±{a['w05'][1]:.2f} "
              f"over={a['over'][0]:.1f} drops={a['drops'][0]:.1f} t={a['t_min'][0]:.2f}±{a['t_min'][1]:.2f} "
              f"probes={a['probes'][0]:.1f}  " + (f"dt={vs['dt_min_mean']:+.2f}min p={vs['p_time']:.1e} d|e|={vs['dabs_err_mean']:+.3f} p={vs['p_err']:.2f}" if vs else ""))
