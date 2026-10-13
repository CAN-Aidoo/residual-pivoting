"""Breakaway dynamics demo: rod angle during a grip release across the slip threshold,
at several release rates, against the quasi-static equilibrium cos(theta) = F / F_h."""
import json, numpy as np, mujoco
from pivot_env import Rig
o = dict(m=0.0, d=0.0, mu=0.0, mut=0.0)
R = json.load(open("eval_val_guard2.json")); r5 = [r for r in R if r["method"] == "residualG_s2" and r["i"] == 5][0]
o = {k: r5[k] for k in ("m", "d", "mu", "mut")}
Fh = o["m"] * 9.81 * o["d"] / (2 * o["mut"])
out = dict(obj=o, F_h=Fh, runs={})
for rate in (0.05, 0.15, 0.3, 0.5):
    rig = Rig(**o)
    rig.ramp_grip(1.08 * Fh, secs=0.5, hold=0.5)
    F0 = rig.grip; F1 = 0.80 * Fh; dt = rig.model.opt.timestep
    n = int((F0 - F1) / rate / dt); t, F, th = [], [], []
    for k in range(n + int(2.0 / dt)):
        Fk = F0 + (F1 - F0) * min(1.0, (k + 1) / n); rig.data.ctrl[1:] = Fk
        mujoco.mj_step(rig.model, rig.data)
        if k % 10 == 0: t.append(k * dt); F.append(Fk); th.append(rig.com_dir())
    out["runs"][rate] = dict(t=t, F=F, th=th, final=th[-1])
    print(rate, "final", round(th[-1], 3), "quasi-static", round(-np.arccos(F1 / Fh), 3))
out["qs_final"] = float(-np.arccos(0.80))
json.dump(out, open("breakaway.json", "w"))
