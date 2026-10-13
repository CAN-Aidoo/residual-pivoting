"""Write paper2/numbers.tex from result files. Every number in the paper comes from here."""
import json, numpy as np
def fp(p): return "{:.2f}".format(p) if p >= 0.1 else ("{:.3f}".format(p) if p >= 0.001 else "{:.0e}".format(p))
S = json.load(open("results_summary.json")); B = json.load(open("breakaway.json"))
N = {}
def put(k, v, f="{:.3f}"): N[k] = f.format(v)
for sname, tag in [("indist", "I"), ("current", "C"), ("ood", "O"), ("val", "V"), ("confirm", "K")]:
    for pk, pt in [("residualG_vs_modelG", "GvG"), ("pure_vs_residual", "PvR")]:
        pv = S[sname].get("pairs", {}).get(pk)
        if pv:
            put(tag + pt + "dE", pv["dabs_err"], "{:+.3f}"); put(tag + pt + "dT", pv["dt_min"], "{:+.2f}")
            for kk in ("p_err", "p_time"):
                N[tag + pt + ("pE" if kk == "p_err" else "pT")] = fp(pv[kk])
    for fam, ft in [("model", "Mod"), ("modelG", "ModG"), ("pure", "Pur"), ("residual", "Res"), ("residualG", "ResG"), ("residualR", "ResR")]:
        if fam not in S[sname]: continue
        a = S[sname][fam]["mean_sd"]; p = tag + ft
        put(p + "Rms", a["rms"][0]); put(p + "RmsSd", a["rms"][1])
        put(p + "Five", 100 * a["w05"][0], "{:.0f}"); put(p + "FiveSd", 100 * a["w05"][1], "{:.0f}")
        put(p + "Time", a["t_min"][0], "{:.2f}"); put(p + "TimeSd", a["t_min"][1], "{:.2f}")
        put(p + "Probes", a["probes"][0], "{:.1f}"); put(p + "Over", a["over"][0], "{:.1f}"); put(p + "Drops", a["drops"][0], "{:.1f}")
        vs = S[sname][fam].get("vs_model")
        if vs:
            put(p + "dT", vs["dt_min_mean"], "{:+.2f}"); put(p + "pT", vs["p_time"], "{:.0e}")
            put(p + "dE", vs["dabs_err_mean"], "{:+.3f}"); N[p + "pE"] = fp(vs["p_err"])
            put(p + "TimeCut", -100 * vs["dt_min_mean"] / S[sname]["model"]["mean_sd"]["t_min"][0], "{:.0f}")
        for m, d in S[sname][fam]["per_seed"].items():
            sd = m[-1] if m[-2] == "s" else ""
            put(p + "Rms" + "abc"[int(sd)] if sd else p + "RmsOnly", d["rms"])
            if sd: put(p + "Five" + "abc"[int(sd)], 100 * d["w05"], "{:.0f}")
for r, run in B["runs"].items():
    k = {"0.05": "a", "0.15": "b", "0.3": "c", "0.5": "d"}[r]
    put("Brk" + k, run["final"]); put("BrkOver" + k, abs(run["final"] - B["qs_final"]))
put("BrkQs", B["qs_final"])
SS = json.load(open("search_stats.json"))
put("SrchModStep", 100 * SS["model"]["step"], "{:.0f}")
for s_ in range(3):
    put("SrchStep" + "abc"[s_], 100 * SS[f"residual_s{s_}"]["step"], "{:.0f}"); put("SrchRate" + "abc"[s_], SS[f"residual_s{s_}"]["rate"], "{:.2f}")
put("ObjSevenMod", SS["obj7"]["model"]); put("ObjSevenModG", SS["obj7"]["modelG"]); put("ObjSevenTgt", SS["obj7"]["target"])
logs = [json.load(open(f"log_{m}_s{s}.json")) for m in ("residual", "pure") for s in range(3)]
put("EpPerRun", logs[0][-1]["episodes"], "{:d}")
FS = json.load(open("firstslip_stats.json"))
put("FsRot", FS["rot_med"], "{:.2f}"); put("FsRhoStep", FS["rho_step"], "{:.2f}"); put("FsRhoRate", FS["rho_rate"], "{:.2f}")
put("FsExceed", 100 * FS["exceed"], "{:.0f}")
for k, v in FS.get("extra", {}).items(): N[k] = v
open("/home/claude/paper2/numbers.tex", "w").write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(N.items())))
print(len(N), "macros"); print({k: N[k] for k in sorted(N) if k.startswith(("IMod", "IRes", "CRes", "CMod", "Brk"))})
