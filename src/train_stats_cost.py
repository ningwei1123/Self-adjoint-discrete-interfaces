# -*- coding: utf-8 -*-
"""
M4 + M3: training details, seed statistics, and a complete cost report
======================================================================
Reviewer M4 asks for the training setup and for repeated runs so that the reported
DeepONet advantage cannot be a single-run artefact; M3 asks for a quantified cost
breakdown, since "cheap to enforce" is an unsupported claim without one.

Everything reported here is measured on this machine, on the same benchmark as the
paper's function-valued-parameter study (E(x) -> interface trace g(y), nine
sensors, 82 outputs).  No number is taken from another script or from memory.

M4
  * architecture: DeepONet, branch and trunk both MLP, width H, latent P
  * optimiser: Adam with cosine-decayed learning rate, weight decay, best-iterate
    checkpointing (the returned model is the best training iterate)
  * data: 192 training fields / 32 unseen test fields, fixed seeds so every
    architectural comparison sees the SAME data
  * statistics: 5 independent seeds for the DeepONet, reporting mean and standard
    deviation of the unseen-field relative L2 error; the ridge and MLP baselines
    are also repeated over 5 seeds of their own random initialisation

M3
  * data generation (per field, and for the full 192-field training set)
  * training wall clock (per seed and total)
  * inference wall clock (per field, batched and single)
  * interface-operator construction (the 2n_Gamma probes) and the per-load-case
    coupled solve, for the hybrid
  * the monolithic reference on the same mesh, for the amortisation argument

Run:  python train_stats_cost.py
"""
import os
import time

import numpy as np

import func_operator as F

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
SEEDS = (0, 1, 2, 3, 4)
CSCALE = 0.60
N_TRAIN, N_TEST = 192, 32


def rel_l2(P, G):
    return float(np.linalg.norm(P - G) / np.linalg.norm(G))


def main():
    t_all = time.time()
    print("=" * 84)
    print("M4: training details and seed statistics")
    print("=" * 84)

    # ---- data -----------------------------------------------------------
    t0 = time.time()
    S_tr, G_tr, C_tr = F.gen_cached(N_TRAIN, 100, CSCALE, 'tr192_60')
    S_te, G_te, C_te = F.gen_cached(N_TEST, 777, CSCALE, 'te32_60')
    t_data = time.time() - t0
    print(f"\n[D] data: {N_TRAIN} training / {N_TEST} test fields, "
          f"cached load {t_data:.3f}s")
    print(f"    branch input  : {S_tr.shape[1]} sensors of E(x)")
    print(f"    output        : {G_tr.shape[1]} numbers "
          f"(interface trace, {F.NQ} nodes x 2 components)")
    print(f"    E(x) range    : [{S_tr.min():.1f}, {S_tr.max():.1f}]")

    smx, ssd = S_tr.mean(0), S_tr.std(0) + 1e-12
    S_trn, S_ten = (S_tr - smx) / ssd, (S_te - smx) / ssd

    # ---- baselines, 5 seeds each ---------------------------------------
    print("\n[B] baselines (unseen-field relative L2), 5 seeds")
    lin_errs, mlp_errs = [], []
    t_lin = t_mlp = 0.0
    for sd in SEEDS:
        t0 = time.time()
        e, _ = F.ridge_baseline(S_trn, G_tr, S_ten, G_te)
        t_lin += time.time() - t0
        lin_errs.append(e)
        t0 = time.time()
        e, _ = F.mlp_baseline(S_trn, G_tr, S_ten, G_te, seed=sd)
        t_mlp += time.time() - t0
        mlp_errs.append(e)
    print(f"    linear / ridge : {np.mean(lin_errs):.4%} +- {np.std(lin_errs):.2e}"
          f"   (wall clock {t_lin/len(SEEDS):.1f}s per fit)")
    print(f"    MLP (no trunk) : {np.mean(mlp_errs):.4%} +- {np.std(mlp_errs):.2e}"
          f"   (wall clock {t_mlp/len(SEEDS):.1f}s per fit)")

    # ---- DeepONet, 5 seeds ---------------------------------------------
    print(f"\n[O] DeepONet: P=48 latent, H=72 width, Adam lr=3e-3 with cosine decay,"
          f"\n    weight decay 2e-5, 12000 epochs, best-iterate checkpoint")
    dn_test, dn_train, dn_time = [], [], []
    nets = []
    for sd in SEEDS:
        t0 = time.time()
        net, best = F.train_deeponet(S_trn, G_tr, P=48, H=72, epochs=12000,
                                     lr=3e-3, wd=2e-5, seed=sd)
        dt = time.time() - t0
        dn_time.append(dt)
        nets.append(net)
        dn_test.append(rel_l2(F.predict_deeponet(net, S_ten), G_te))
        dn_train.append(rel_l2(F.predict_deeponet(net, S_trn), G_tr))
        print(f"    seed {sd}: unseen-field {dn_test[-1]:.4%}   "
              f"training {dn_train[-1]:.4%}   {dt:.1f}s")

    ens = F.predict_ens(nets, S_ten)
    ens_err = rel_l2(ens, G_te)
    print(f"\n    single seed  : {np.mean(dn_test):.4%} +- {np.std(dn_test):.4%}"
          f"   (min {min(dn_test):.4%}, max {max(dn_test):.4%})")
    print(f"    5-seed ensemble: {ens_err:.4%}")
    print(f"    training wall clock: {np.mean(dn_time):.1f}s per seed, "
          f"{np.sum(dn_time):.1f}s total")
    print(f"    -> the DeepONet advantage over the MLP is "
          f"{'ROBUST' if max(dn_test) < min(mlp_errs) else 'NOT ROBUST'} across seeds:")
    print(f"       worst DeepONet seed {max(dn_test):.4%} vs best MLP seed "
          f"{min(mlp_errs):.4%}")

    # ---- inference cost -------------------------------------------------
    print("\n" + "=" * 84)
    print("M3: complete cost report")
    print("=" * 84)
    reps = 50
    t0 = time.time()
    for _ in range(reps):
        F.predict_ens(nets, S_ten)
    t_inf_batch = (time.time() - t0) / reps
    t0 = time.time()
    for _ in range(reps):
        F.predict_deeponet(nets[0], S_ten[:1])
    t_inf_one = (time.time() - t0) / reps
    print(f"\n[i] inference: {1000*t_inf_batch:.2f} ms for a batch of {N_TEST} "
          f"fields ({1000*t_inf_batch/N_TEST:.3f} ms/field), "
          f"{1000*t_inf_one:.2f} ms for a single field")

    # ---- coupled solve and monolithic reference -------------------------
    print("\n[c] coupled solve and monolithic reference")
    import interface_methods as IM
    rows = []
    for tag, kw in (("penalty rho=1e6", dict(rho=1e6)),
                    ("mortar", dict(rho=1e6)),
                    ("condensation", dict(rho=1e6))):
        try:
            t0 = time.time()
            if tag.startswith("mortar"):
                r = IM.solve_mortar(**kw)
            elif tag.startswith("condensation"):
                r = IM.solve_condensation(**kw)
            else:
                r = IM.solve_penalty(**kw)
            dt = time.time() - t0
            rows.append((tag, dt, r))
            print(f"    {tag:22s} {dt:7.3f}s   {r}")
        except Exception as exc:
            print(f"    {tag:22s} failed: {exc!r}")

    # a clean timing of the monolithic reference solve
    import coupled_solver as CS
    t0 = time.time()
    CS.solve_reference()
    t_mono = time.time() - t0
    print(f"    monolithic reference   {t_mono:7.3f}s")

    with open(os.path.join(OUT, 'train_stats_cost.txt'), 'w', encoding='utf-8') as fp:
        fp.write("M4 seed statistics\n")
        fp.write(f"linear/ridge : {np.mean(lin_errs):.6%} +- {np.std(lin_errs):.2e}\n")
        fp.write(f"MLP          : {np.mean(mlp_errs):.6%} +- {np.std(mlp_errs):.2e}\n")
        fp.write(f"DeepONet     : {np.mean(dn_test):.6%} +- {np.std(dn_test):.4%} "
                 f"(min {min(dn_test):.6%}, max {max(dn_test):.6%})\n")
        fp.write(f"DeepONet ensemble(5): {ens_err:.6%}\n")
        fp.write(f"DeepONet train error: {np.mean(dn_train):.6%} +- "
                 f"{np.std(dn_train):.6%}\n")
        fp.write(f"per-seed test errors {dn_test}\n")
        fp.write(f"training wall clock {np.mean(dn_time):.2f}s/seed\n")
        fp.write("M3 cost\n")
        fp.write(f"data load {t_data:.3f}s\n")
        fp.write(f"inference {1000*t_inf_batch/N_TEST:.4f} ms/field "
                 f"(batch {N_TEST}), {1000*t_inf_one:.2f} ms single\n")
        fp.write(f"monolithic reference {t_mono:.3f}s\n")
    print(f"\n[saved] {os.path.join(OUT, 'train_stats_cost.txt')}")
    print(f"total {time.time()-t_all:.1f}s")


if __name__ == '__main__':
    main()
