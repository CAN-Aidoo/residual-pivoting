"""Extra statistics quoted in the paper text (written into firstslip_stats.json['extra'] as LaTeX macros)."""
import json, numpy as np
R = json.load(open("traj_indist.json")); FS = json.load(open("firstslip_stats.json"))
T = {r["i"]: r["target"] for r in json.load(open("eval_indist.json")) if r["method"] == "model"}
S = json.load(open("results_summary.json"))
fams = {"Mod": ["model"], "Res": [f"residual_s{s}" for s in range(3)], "ResG": [f"residualG_s{s}" for s in range(3)], "Pur": [f"pure_s{s}" for s in range(3)]}
E = {}
for f, ms in fams.items():
    rr = [r for r in R if r["method"] in ms]
    over = [r for r in rr if r["held"] and r["err"] < -0.05]
    # pre = steps before the first-slip step; post = steps after it (the slip step itself excluded)
    E[f"Pre{f}"] = f"{np.mean([r['pre'] for r in rr]):.1f}"; E[f"Post{f}"] = f"{np.mean([r['n'] - r['pre'] - (1 if r['first'] else 0) for r in rr]):.1f}"
    E[f"OverFirst{f}"] = f"{sum(1 for r in over if r['first'] and r['first']['past'] < -0.05) / len(ms):.1f}"
    E[f"OverAll{f}"] = f"{len(over) / len(ms):.1f}"
X = [r for r in R if r["first"] and r["first"]["step"] is not None]
tgt = np.array([-T[r["i"]] for r in X]); of = np.array([r["first"]["past"] < -0.05 for r in X])
E["OvShallow"] = f"{100 * FS['ov_shallow']:.0f}"; E["OvMid"] = f"{100 * FS['ov_mid']:.0f}"; E["OvDeep"] = f"{100 * FS['ov_deep']:.0f}"
E["OvShallowObj"] = str(FS["ov_shallow_n_obj"]); E["FsN"] = str(FS["n_episodes"]); E["FsNobj"] = str(FS["n_objects"])
E["FsPrhoStep"] = f"{FS['prho_step']:.2f}"; E["FsPrhoRate"] = f"{FS['prho_rate']:.2f}"; E["FsRhoSR"] = f"{FS['rho_step_rate']:.2f}"
for f_ in ("Mod", "Res", "ResG", "Pur"):
    E["FsExceed" + f_] = f"{100 * FS['exceed_by_fam'][f_]:.0f}"; E["FsRot" + f_] = f"{FS['rot_med_by_fam'][f_]:.2f}"
E["NumUnderIndist"] = str(sum(r["held"] and r["err"] > 0.05 for r in json.load(open("eval_indist.json")) + json.load(open("eval_indist_guard.json"))))
E["RetMod"] = f"{S['indist']['model']['mean_sd']['score'][0]:.2f}"; E["RetRes"] = f"{S['indist']['residual']['mean_sd']['score'][0]:.2f}"
E["RetResc"] = f"{S['indist']['residual']['per_seed']['residual_s2']['score']:.2f}"
L = json.load(open("log_residual_s2.json")); E["TrResFirst"] = f"{np.mean([l['f_center'] for l in L[:5]]):.2f}"; E["TrResLast"] = f"{np.mean([l['f_center'] for l in L[-5:]]):.2f}"
FS["extra"] = E; json.dump(FS, open("firstslip_stats.json", "w"), indent=1); print(E)
