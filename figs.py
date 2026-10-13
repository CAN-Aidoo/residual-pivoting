import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#8a8a86", "axes.labelcolor": "#3b3b38",
                     "xtick.color": "#5c5c58", "ytick.color": "#5c5c58", "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("results_summary.json"))
COL = {"model": "#5c5c58", "modelG": "#9a9994", "pure": "#eb6834", "residual": "#86b6ef", "residualG": "#2a78d6"}
LBL = {"model": "Model controller", "modelG": "Model + guard", "pure": "Pure RL", "residual": "Residual RL", "residualG": "Residual RL + guard"}
MK = {"model": "s", "modelG": "D", "pure": "^", "residual": "o", "residualG": "o"}

# Fig 1: accuracy vs time, per seed, two sensing conditions
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=200)
for ax, (sname, title) in zip(axs, [("indist", "Noise-model sensing (40 objects)"), ("current", "Motor-current sensing (20 objects)")]):
    for fam in ["model", "modelG", "pure", "residual", "residualG"]:
        d = S[sname][fam]
        xs = [p["t_min"] for p in d["per_seed"].values()]; ys = [1 - p["w05"] for p in d["per_seed"].values()]
        ax.scatter(xs, ys, s=38, marker=MK[fam], color=COL[fam], edgecolor="white", linewidth=0.8, label=LBL[fam], zorder=3)
    ax.set_title(title, fontsize=9, color="#3b3b38"); ax.set_xlabel("Mean time per pivot (simulated min)")
    ax.grid(color="#ebeae6", lw=0.8); ax.set_ylim(-0.02, 0.33)
axs[0].set_ylabel("Fraction of objects missing\nthe target by > 0.05 rad")
axs[0].legend(frameon=False, fontsize=7.3, loc="upper right")
fig.tight_layout(); fig.savefig("/home/claude/paper2/fig_tradeoff.png")

# Fig 2: learning curves
fig, ax = plt.subplots(figsize=(5.4, 3.1), dpi=200)
for mode, c in [("residual", "#2a78d6"), ("pure", "#eb6834")]:
    for s in range(3):
        L = json.load(open(f"log_{mode}_s{s}.json"))
        x = np.array([l["episodes"] for l in L]) / 1000; y = np.array([l["f_center"] for l in L])
        ys = np.convolve(y, np.ones(4) / 4, mode="valid")
        ax.plot(x[3:], ys, color=c, lw=1.6, alpha=0.85, label=("Residual RL" if mode == "residual" else "Pure RL") if s == 0 else None)
ax.set_xlabel("Training episodes (thousands)"); ax.set_ylabel("Return of current policy on that\niteration's 6 objects (4-iter. avg.)")
ax.grid(color="#ebeae6", lw=0.8); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("/home/claude/paper2/fig_learning.png")

# Fig 3: breakaway dynamics
B = json.load(open("breakaway.json"))
fig, ax = plt.subplots(figsize=(5.4, 3.1), dpi=200)
cs = {"0.05": "#86b6ef", "0.15": "#3987e5", "0.3": "#1c5cab", "0.5": "#eb6834"}
for rate, run in B["runs"].items():
    F = np.array(run["F"]) / B["F_h"]
    ax.plot(F, run["th"], color=cs[rate], lw=1.8, label=f"{float(rate):g} N/s")
Fq = np.linspace(0.8, 1.0, 100); ax.plot(Fq, -np.arccos(Fq), color="#3b3b38", ls="--", lw=1.2, label="quasi-static  cos θ = F/F$_h$")
ax.invert_xaxis(); ax.set_xlabel("Grip force F / F$_h$ (release proceeds left to right)")
ax.set_ylabel("COM direction θ (rad)"); ax.grid(color="#ebeae6", lw=0.8)
ax.legend(title="Release rate", frameon=False, fontsize=7.5, title_fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig("/home/claude/paper2/fig_breakaway.png")
print("ok")
