# residual-pivoting

Code and data for the preprint **"Residual Learning on a Physics Pivoting Controller: Faster Search, and Where the Quasi-Static Model Breaks"** (Clement Aidoo, CANLABS, 2026).

Companion to [Gravity Pivoting from Motor Current](https://doi.org/10.31224/8492) (engrXiv). A learned residual (linear policy, evolution strategies) is added on top of the motor-current gravity-pivoting controller from [motor-current-pivoting](https://doi.org/10.5281/zenodo.23264189), and compared with the plain controller and with a policy learned without the model. **All results are from MuJoCo simulation.** Read the paper's Limitations and "Chronology and possible leakage" paragraphs before reusing any number.

## Requirements
Python 3.12, `mujoco==3.15.0`, `numpy`, `scipy`, `matplotlib`. This repo imports the testbed (`mj_env.py`, `mj_ctrl.py`, `motor_model.py`, `exp_c.py`) from the paper-1 repository: clone it next to this one, or set `PIVOT1_DIR` to its path.

## Files
| File | Purpose |
|---|---|
| `pivot_env.py` | Episodic pivoting environment: model controller, residual / pure / guarded modes, reward |
| `train_es.py` | Antithetic ES training, resumable (`python train_es.py residual 0 20`) |
| `evaluate.py` | Held-out evaluation (`indist`, `ood`, `current`, `val`, `confirm`) |
| `analyze.py` | Aggregation and paired Wilcoxon tests → `results/results_summary.json` |
| `traj_analysis.py`, `firstslip_analysis.py`, `extra_stats.py`, `search_stats.py` | First-slip and search-phase analyses |
| `breakaway.py` | Controlled release-rate experiment (Fig. 3) |
| `figs.py`, `gen_numbers.py` | Figures and every number in the paper (`paper/numbers.tex`) |
| `results/` | Raw per-episode results, training logs, trained weights `w_*.npy` |

## Reproduce (about 6 h CPU for training, 1 h for evaluation, 2 cores)
```bash
for s in 0 1 2; do for m in residual pure; do python train_es.py $m $s 20; done; done
python evaluate.py indist 40; ONLY=modelG,residualG TAG=indist_guard python evaluate.py indist 40
ONLY=model,residual_,residualG TAG=ood_a python evaluate.py ood 30; ONLY=pure TAG=ood_b python evaluate.py ood 30
ONLY=model,residual_,residualG TAG=current_a python evaluate.py current 20; ONLY=pure TAG=current_b python evaluate.py current 20
ONLY=model,residual python evaluate.py val 24; ONLY=residualR TAG=val_rate python evaluate.py val 24
ONLY=model,residual_,residualG python evaluate.py confirm 40   # fresh confirmation set, run once after the guard was frozen
python analyze.py && python traj_analysis.py && python firstslip_analysis.py && python search_stats.py && python breakaway.py
python extra_stats.py && python gen_numbers.py && python figs.py
```
Scripts write outputs to the working directory; the committed `results/` contain the exact files used in the paper.

## AI assistance
An AI assistant (Claude, Anthropic) was used to write and debug code and to draft the paper. The author directed the study, reviewed code, results and text, and is responsible for the content.

## Licence
Code: MIT. Data in `results/` and figures: CC BY 4.0.
