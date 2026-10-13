"""Antithetic evolution strategies for the pivoting policies (residual vs pure).

Usage: python train_es.py MODE SEED [ITERS]
Identical budget, observation, objects and seeds for both modes.
"""
import sys, os, json, time, pickle, numpy as np
from multiprocessing import Pool
from pivot_env import PivotEpisode, sample_object, score, LinearPolicy, OBS_DIM

POP_PAIRS = 6; N_OBJ = 6; SIGMA = 0.1; LR = 0.06; NPROC = 2


def evaluate(args):
    mode, w, objs, seeds = args
    pol = LinearPolicy(w)
    return float(np.mean([score(PivotEpisode(o, np.random.default_rng(s)).run(mode, pol))
                          for o, s in zip(objs, seeds)]))


def main():
    mode, seed = sys.argv[1], int(sys.argv[2]); iters = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    ck = f"ckpt_{mode}_s{seed}.pkl"
    if os.path.exists(ck):
        st = pickle.load(open(ck, "rb")); rng, w, m, v, log, start = st["rng"], st["w"], st["m"], st["v"], st["log"], st["it"] + 1
    else:
        rng = np.random.default_rng(seed)
        w = np.zeros(2 * OBS_DIM); m = np.zeros_like(w); v = np.zeros_like(w); log = []; start = 0
    t0 = time.time() - (log[-1]["wall"] if log else 0)
    with Pool(NPROC) as pool:
        for it in range(start, iters):
            if time.time() > float(os.environ.get('DEADLINE', '1e12')):
                sys.exit(3)
            objs = [sample_object(rng) for _ in range(N_OBJ)]
            seeds = [int(x) for x in rng.integers(0, 2**31, N_OBJ)]
            eps = rng.normal(size=(POP_PAIRS, w.size))
            jobs = [(mode, w + SIGMA * e, objs, seeds) for e in eps] + [(mode, w - SIGMA * e, objs, seeds) for e in eps]
            jobs.append((mode, w, objs, seeds))
            f = np.array(pool.map(evaluate, jobs))
            fp, fm, f0 = f[:POP_PAIRS], f[POP_PAIRS:2 * POP_PAIRS], f[-1]
            allf = np.concatenate([fp, fm]); ranks = allf.argsort().argsort() / (allf.size - 1) - 0.5
            rp, rm = ranks[:POP_PAIRS], ranks[POP_PAIRS:]
            g = ((rp - rm)[:, None] * eps).sum(0) / (POP_PAIRS * SIGMA)
            # Adam ascent
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            w = w + LR * (m / (1 - 0.9 ** (it + 1))) / (np.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
            log.append(dict(it=it, f_center=float(f0), f_mean=float(allf.mean()), f_best=float(allf.max()),
                            episodes=(it + 1) * len(jobs) * N_OBJ, wall=time.time() - t0))
            print(mode, seed, it, round(f0, 3), round(allf.mean(), 3), round(time.time() - t0), flush=True)
            np.save(f"w_{mode}_s{seed}.npy", w)
            json.dump(log, open(f"log_{mode}_s{seed}.json", "w"), indent=1)
            pickle.dump(dict(rng=rng, w=w, m=m, v=v, log=log, it=it), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)


if __name__ == "__main__":
    main()
