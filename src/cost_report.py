# -*- coding: utf-8 -*-
"""
M3 (final): end-to-end cost accounting for the hybrid solver
===========================================================
This file replaces an earlier version that reported a surrogate error of 19.05%
while every isolated re-measurement of the same quantity gave 1.43%.  The cause
was never traced, and the only defensible response is to remove the state that
made the two disagree: this version GENERATES THE DATA IT TRAINS ON in the same
call, so no on-disk cache and no other module's globals can influence it.  It also
reports the accuracy of every network individually next to the ensemble, so a
future disagreement of that kind is visible immediately instead of silently
replacing the headline number.

Only wall-clock times are reported; all are measured in this run on one core.

Run:  python cost_report.py
"""
import os
import time

import numpy as np

import func_operator as F
import interface_methods as IM
import coupled_solver as CS

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
CSCALE = 0.60


def rel_l2(P, G):
    return float(np.linalg.norm(P - G) / np.linalg.norm(G))


def main():
    rows = []
    print("=" * 84)
    print("M3: end-to-end cost accounting (single core, wall clock)")
    print("=" * 84)

    # ---- training-data generation, measured uncached ---------------------
    print("\n[1] training-data generation (uncached: one interface solve per field)")
    t0 = time.time()
    F.gen(1, 111, CSCALE)
    t1 = time.time() - t0
    n = 16
    t0 = time.time()
    F.gen(n, 222, CSCALE)
    t16 = time.time() - t0
    gen_per = t16 / n
    print(f"    1 field                 : {t1:.3f}s")
    print(f"    {n} fields                : {t16:.3f}s  ->  {gen_per:.3f}s/field, "
          f"{gen_per*192:.1f}s for the 192-field training set")
    rows.append(("data generation, per field", gen_per))
    rows.append(("data generation, 192 fields", gen_per * 192))

    # ---- monolithic reference -------------------------------------------
    print("\n[2] monolithic FE reference")
    t0 = time.time()
    ref = CS.solve_reference()
    t_ref = time.time() - t0
    nn = len(ref['nodes']) if isinstance(ref, dict) and 'nodes' in ref else '?'
    print(f"    monolithic solve        : {t_ref:.3f}s   ({nn} nodes)")
    rows.append(("monolithic reference", t_ref))

    # ---- interface schemes ----------------------------------------------
    print("\n[3] coupled interface schemes (setup + solve, one load case)")
    for fn, tag in ((IM.solve_penalty, "penalty rho=1e6"),
                    (IM.solve_mortar, "mortar"),
                    (IM.solve_condensation, "static condensation")):
        t0 = time.time()
        r = fn(rho=1e6)
        dt = time.time() - t0
        N = r.get('N') if isinstance(r, dict) else None
        print(f"    {tag:24s} {dt:7.3f}s" + (f"   dofs {N}" if N else ""))
        rows.append((tag, dt))

    # ---- surrogate: fresh data, per-seed AND ensemble accuracy -----------
    print("\n[4] surrogate accuracy and cost (data generated in this call)")
    t0 = time.time()
    S_tr, G_tr, _ = F.gen(192, 100, CSCALE)
    S_te, G_te, _ = F.gen(32, 777, CSCALE)
    t_gen_full = time.time() - t0
    print(f"    data: 192 train / 32 test, generated in {t_gen_full:.1f}s")
    smx, ssd = S_tr.mean(0), S_tr.std(0) + 1e-12
    S_trn, S_ten = (S_tr - smx) / ssd, (S_te - smx) / ssd

    t0 = time.time()
    nets = []
    for sd in (0, 1, 2):
        net, _ = F.train_deeponet(S_trn, G_tr, P=48, H=72, epochs=12000,
                                  lr=3e-3, wd=2e-5, seed=sd)
        nets.append(net)
        print(f"    seed {sd}: unseen-field "
              f"{rel_l2(F.predict_deeponet(net, S_ten), G_te):.4%}")
    t_train3 = time.time() - t0
    print(f"    3-seed training         : {t_train3:.1f}s "
          f"({t_train3/3:.1f}s per seed)")
    rows.append(("surrogate training, 3 seeds", t_train3))

    # ACCURACY IS NOT REPORTED HERE.  An earlier version of this file printed a
    # surrogate error of 19.05% while every independent re-measurement of the same
    # quantity gave 1.43%.  The cause was never established.  Ruled out, with
    # measurements: the training/test cache matches its documented seeds exactly
    # (np.allclose against a fresh generation); a one-row mis-pairing of S and G
    # would give 28%, and a different generation seed 29-31%, neither of which is
    # 19%; scoring on the extrapolation set (cscale 0.75) gives 6.33%; using the
    # normalised target would give ~100%.  None reproduces it.
    #
    # Rather than publish a figure that cannot be reproduced, this file now reports
    # ONLY the wall-clock costs, which is its purpose, and the surrogate accuracy is
    # taken from train_stats_cost.py, where it is measured over five seeds as
    # 1.8914% +- 0.2889% (single seeds) and 1.3051% (ensemble).  A self-check is kept
    # so that any future disagreement between an ensemble and its members is visible
    # instead of silently replacing the headline number.
    P_ens = F.predict_ens(nets, S_ten)
    singles = [rel_l2(F.predict_deeponet(nt, S_ten), G_te) for nt in nets]
    err_ens = rel_l2(P_ens, G_te)
    print(f"    (accuracy cross-check only, not reported: members "
          f"{', '.join(f'{e:.4%}' for e in singles)}; ensemble {err_ens:.4%})")
    if err_ens > max(singles):
        print("    *** WARNING: ensemble worse than every member -- investigate ***")

    reps = 200
    t0 = time.time()
    for _ in range(reps):
        F.predict_ens(nets, S_ten)
    t_batch = (time.time() - t0) / reps
    per_case = t_batch / 32
    print(f"    inference, batch of 32  : {1000*t_batch:.3f} ms "
          f"({1000*per_case:.4f} ms/field)")
    rows.append(("inference, 32 fields", t_batch))
    rows.append(("inference, single field", per_case))

    # ---- amortisation ----------------------------------------------------
    print("\n[5] amortisation (the honest form of the cost claim)")
    print(f"    monolithic re-solve     : {t_ref*1000:9.1f} ms per load case")
    print(f"    surrogate inference     : {per_case*1000:9.4f} ms per load case")
    print(f"    one-off training        : {t_train3:9.1f} ms (3 seeds) "
          f"+ {t_gen_full*1000:.0f} ms data")
    saving = t_ref - per_case
    print(f"    break-even              : {(t_train3+t_gen_full)/saving:.0f} load "
          f"cases before the surrogate pays for itself")

    with open(os.path.join(OUT, 'cost_report.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M3 end-to-end cost accounting (wall clock, single core)\n")
        for tag, dt in rows:
            fp.write(f"{tag:34s} {dt:.6f}s\n")
        fp.write("surrogate accuracy: NOT reported here; see train_stats_cost.py "
                 "(5 seeds)\n")
        fp.write(f"amortisation {t_ref/per_case:.0f}x per load case; "
                 f"break-even {(t_train3+t_gen_full)/saving:.0f} load cases\n")
    print(f"\n[saved] {os.path.join(OUT, 'cost_report.txt')}")


if __name__ == '__main__':
    main()
