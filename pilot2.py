import time, json, sys, numpy as np
from pivot_env import PivotEpisode, sample_object, score
g = np.random.default_rng(5); objs = [sample_object(g) for _ in range(int(sys.argv[1]))]
rows = []; t0 = time.time()
for i, o in enumerate(objs):
    res = PivotEpisode(o, np.random.default_rng(100 + i)).run("model", log=True)
    st = len(res["traj"]); srch = sum(1 for k, F, rt, th, tc in res["traj"] if th > -0.03 + 0)  # rough
    rows.append(dict(i=i, err=res["err"], held=res["held"], t=res["t_sim"], steps=st, score=score(res), mut=o["mut"]))
    print(i, round(res["err"], 3), res["held"], round(res["t_sim"] / 60, 2), st, round(o["mut"], 4), flush=True)
e = np.array([r["err"] for r in rows if r["held"]])
print("rms", np.sqrt(np.mean(e**2)), "w05", np.mean(np.abs(e) <= .05), "tmin", np.mean([r["t"] for r in rows]) / 60,
      "steps", np.mean([r["steps"] for r in rows]), "score", np.mean([r["score"] for r in rows]), "wall/ep", (time.time() - t0) / len(rows))
