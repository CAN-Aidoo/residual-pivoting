"""Episodic pivoting environment for residual RL (paper 2).

Same MuJoCo testbed as paper 1 (mj_env / mj_ctrl). One episode = pivot one randomized
object from a horizontal clamped grasp to a target centre-of-mass angle.

Each decision step: (optional fast pre-ramp to 1.05*F_last) -> slow release to F at
`rate` N/s -> hold -> smooth clamp -> measure angle.  The paper-1 controller supplies a
model-based proposal (F_model, rate0).  A policy acts on top of it:

  mode="model"    : F = F_model,                  rate = rate0             (paper 1)
  mode="residual" : F = F_model * exp(sF*a0),     rate = rate0*exp(sR*a1)
  mode="pure"     : F = F_last  * exp(-sP*(1+a0)/2 ... ) i.e. learned step from scratch,
                    rate = rate0*exp(sR*a1)

Measurement during training: true COM direction + Gaussian noise (sigma_meas), a sensor
surrogate calibrated on paper 1 (self-estimate RMS 0.008 rad). `meas="current"` swaps in
the full motor-current probe + least-squares pipeline from paper 1 for final evaluation.
"""
import sys, os
sys.path.insert(0, os.environ.get("PIVOT1_DIR", os.path.join(os.path.dirname(__file__), "..", "motor-current-pivoting")))
import numpy as np
from mj_ctrl import Rig

TOL = 0.03          # stop band above target (rad), as paper 1
SEARCH = 0.94       # grip search factor before motion is seen
GAIN = 0.7          # fraction of remaining angle per model step
MUT_MIN = 0.002
RATE0 = 0.3         # paper-1 release rate (N/s)
MAX_STEPS = 30
SF, SR = 0.25, 1.4  # residual scales: F x[0.78,1.28], rate x[0.25,4.1]
SP = 0.5            # pure policy step range


def sample_object(rng, ood=False):
    if not ood:
        return dict(m=rng.uniform(0.05, 0.2), d=rng.uniform(0.02, 0.06), mu=rng.uniform(0.4, 0.8),
                    mut=rng.uniform(0.002, 0.006), target=rng.uniform(-1.3, -0.3))
    # out-of-range: heavier, longer offset, higher torsional friction, wider targets, but still
    # holdable by the 30 N clamp (required grip F_h = m g d / (2 mu_t) <= 25 N); rejection-sampled
    while True:
        o = dict(m=rng.uniform(0.2, 0.3), d=rng.uniform(0.06, 0.08), mu=rng.uniform(0.3, 0.9),
                 mut=rng.uniform(0.004, 0.008), target=rng.uniform(-1.4, -0.2))
        if o["m"] * 9.81 * o["d"] / (2 * o["mut"]) <= 25.0:
            return o


def features(th, target, F_last, F0, k, moved, c_hat):
    """Small, scale-free observation vector shared by all learned policies."""
    rem = th - target
    return np.array([rem, th, target, np.log(max(F_last, 1e-3) / F0), k / MAX_STEPS,
                     float(moved), np.cos(th) - c_hat * F_last if moved else 0.0, 1.0])


OBS_DIM = 8


class PivotEpisode:
    def __init__(self, obj, rng, sigma_meas=0.008, meas="surrogate"):
        self.o = obj; self.rng = rng; self.sig = sigma_meas; self.meas_mode = meas
        self.r = Rig(obj["m"], obj["d"], obj["mu"], obj["mut"])
        if meas == "current":
            try:
                from exp_c2 import initial_probe, KT_NOM
            except ImportError:  # named exp_c.py in the paper-1 repository
                from exp_c import initial_probe, KT_NOM
            self._probe = initial_probe
            _, A_amp = initial_probe(self.r, rng)
            self.A_est = A_amp * KT_NOM * rng.uniform(0.9, 1.1)
        else:
            self.A_est = obj["m"] * 9.81 * obj["d"] * rng.uniform(0.9, 1.1)
        self.probes = 1

    def measure(self):
        self.probes += 1
        if self.meas_mode == "current":
            return float(self._probe(self.r, self.rng)[0])
        return self.r.com_dir() + self.rng.normal(0, self.sig)

    def run(self, mode="model", policy=None, log=False):
        r, o = self.r, self.o
        target = o["target"]
        th0 = self.measure(); th = th0
        F0 = min(30.0, self.A_est / (2 * MUT_MIN)); F = F0
        F_last = None; moved = False; c_hat = 0.0
        traj = []
        for k in range(MAX_STEPS):
            if not r.held(): break
            # ---- paper-1 model proposal ----
            if F_last is None:
                F_model = F0
            elif not moved:
                F_model = F_last * SEARCH
            else:
                aim = th + GAIN * (target - th)
                F_model = min(F_last * 0.995, np.cos(aim) / max(c_hat, 1e-6))
            rate = RATE0
            if mode not in ("model", "modelG") and F_last is not None:
                a = np.clip(policy(features(th, target, F_last, F0, k, moved, c_hat)), -1, 1)
                rate = RATE0 * np.exp(SR * a[1])
                if mode in ("residual", "residualG"):
                    F = F_model * np.exp(SF * a[0])
                else:  # pure: learned multiplicative step down from the last force
                    F = F_last * np.exp(-SP * (1 + a[0]) / 2)
                F = min(F, F_last * 0.995)
                if mode == "residualG" and moved:
                    # physics guard: once the rod moves, the learned residual may only be
                    # more cautious than the quasi-static model (hold more grip, release slower)
                    F = max(F, F_model); rate = min(rate, RATE0)
            else:
                F = F_model
            if mode == "residualR":
                # rate guard only: the residual may release slower than the model, never faster
                rate = min(rate, RATE0)
            if mode in ("residualG", "modelG") and F_last is not None and not moved:
                # search-phase bound: no motion at F_last => F_h <= F_last, so releasing to F
                # rotates the rod by at most acos(F/F_last); never step past the target
                F = max(F, F_last * np.cos(target))
            F = float(np.clip(F, 0.3, 30.0))
            # ---- execute: fast pre-ramp, slow release, hold, clamp, measure ----
            if F_last is not None and 1.05 * F_last < r.grip:
                r.ramp_grip(1.05 * F_last, secs=0.5, hold=0.0)
            r.ramp_grip(F, secs=max(0.5, (r.grip - F) / rate), hold=1.0)
            F_last = F
            r.clamp(); th = self.measure()
            if log: traj.append((k, F, rate, th, r.com_dir()))
            if not moved and th - th0 <= -0.03:
                moved = True
            if moved:
                c_hat = np.cos(th) / F_last
            if th <= target + TOL:
                break
        final = r.com_dir(); held = r.held()
        return dict(final=final, err=final - target, held=held, t_sim=r.t_sim,
                    probes=self.probes, steps=len(traj) if log else None, traj=traj)


def score(res, lam=0.5):
    """Episode return: accuracy in units of 0.05 rad, time in minutes, drop penalty."""
    if not res["held"]:
        return -10.0
    return -abs(res["err"]) / 0.05 - lam * res["t_sim"] / 60.0


class LinearPolicy:
    """a = tanh(W o); W shape (2, OBS_DIM). Zero weights == model controller for residual mode."""
    def __init__(self, w=None):
        self.w = np.zeros((2, OBS_DIM)) if w is None else w.reshape(2, OBS_DIM)

    def __call__(self, o):
        return np.tanh(self.w @ o)
