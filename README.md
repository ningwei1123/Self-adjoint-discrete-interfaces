# Self-adjoint discrete interfaces for coupling finite elements with neural operators

**An exact energy-matching criterion** — code and data for the manuscript.

Wei Ning, School of Mechanical Engineering, Shaanxi University of Technology,
Hanzhong 723001, Shaanxi, China — `ningwei@snut.edu.cn`

**ORCID:** 0009-0006-6563-1892

**Paper DOI:** _to be added on acceptance_

---

## 1. What this project does

The repository contains the code that produces every numerical result and every figure in
the manuscript. The work answers three questions about hybrid solvers that couple a
finite-element subdomain to a network-represented subdomain:

1. **How should the interface be discretised?** The interface trace is carried in the
   nodal trace space with `2 n_t` degrees of freedom. Two ways of extracting the interface
   traction are compared: a **Galerkin pairing**, which forms the traction from the
   consistent nodal reactions, and a **collocation pairing**, which samples it at the
   nodes.

2. **Which invariant actually matters?** Two quantities are often conflated: the
   **energy-matching number** `chi = M_N / M_F`, which governs how the imposed interface
   motion is split between the two subdomains, and the **interface accuracy**. The
   manuscript shows they are independent, and that a *balanced* interface satisfies
   `chi ~ 1`, equivalently `u_Gamma / Delta = 1 / (1 + chi)`.

3. **What does self-adjointness cost?** The Galerkin pairing makes the discrete
   Dirichlet-to-Neumann operator self-adjoint to round-off; the collocation pairing does
   not, and its asymmetry is structured rather than random. This is the mechanism behind
   the accuracy difference.

The repository holds the solvers (`src/`), one script per figure (`scripts/figure_1.py`
… `scripts/figure_9.py`), and the input records the figures need (`data/`). Running a
driver from `src/` writes a plain-text record into `results/`; that directory is created
on first use and is not stored here.

---

## 2. Installation

Python 3.10 or newer. **Tested on Python 3.13.14.**

```bash
git clone <repository-url>
cd <repository-name>
python -m pip install -r requirements.txt
```

`requirements.txt` separates the core dependencies from PyTorch, which is needed **only**
for the learning surrogates (DeepONet, the physics-informed baselines) and for the seed
statistics of Section 5.12. Everything in `src/` and all nine figure scripts run without
it except `figure_8.py`, `src/func_operator.py`, `src/ddpinn_*.py` and
`src/deeponet_*.py`.

### Tested environment

| package | version used |
|---|---|
| Python | 3.13.14 |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |
| SymPy | 1.11+ |
| Pillow | 9.5+ |
| PyTorch | 2.14.1 (CPU) |

No script contains an absolute path; every path is resolved relative to the file that uses
it, so the tree runs from any location.

---

## 3. Reproducing the figures

### 3.1 Figure-to-script map

Run each command **from the repository root**. Output is written to `results/figs/`
(PNG at 300 dpi, with PDF and TIFF masters).

| figure | command | output file in `results/figs/` | what it shows |
|---|---|---|---|
| **Figure 1** | `python scripts/figure_1.py` | `fig01_framework.png` | the coupling framework and its two invariants |
| **Figure 2** | `python scripts/figure_2.py` | `fig02_energy_matching.png` | the two invariants are not the same quantity |
| **Figure 3** | `python scripts/figure_3.py` | `fig03_self_adjoint.png` | the core result: operator defect and spectra |
| **Figure 4** | `python scripts/figure_4.py` | `fig04_consistency.png` | consistency order and the true limiting factor |
| **Figure 5** | `python scripts/figure_5.py` | `fig05_2d_fields.png` | four ways of enforcing one interface formulation |
| **Figure 6** | `python scripts/figure_6.py` | `fig06_3d_multiload.png` | three-dimensional verification, four load cases |
| **Figure 7** | `python scripts/figure_7.py` | `fig07_recovery.png` | why the strong form fails here |
| **Figure 8** | `python scripts/figure_8.py` | `fig08_paramtype.png` | stress recovery and its failure mode |
| **Figure 9** | `python scripts/figure_9.py` | `fig09_neumann.png` | the parameter type selects the architecture |

> Note the file names do not follow the paper order. Figure 3 is the self-adjointness
> figure and Figure 4 the consistency figure, which is the reverse of the file numbering
> because the working order differed from the final presentation order. The table above is
> authoritative; it was verified by running each entry point and recording what it wrote.

All nine entry points run in about two minutes in total on a laptop CPU, with the longest
(Figure 6, the three-dimensional fields) taking under a minute.

### 3.2 Worked example: Figure 4 from a clean checkout

```bash
git clone <repository-url> && cd <repository-name>
python -m pip install -r requirements.txt

# Figure 4 draws the consistency order and reads the recorded arrays written by
# these two drivers:
python src/coupled_solver.py        # 2D: the interface solution and its error
python src/coupled_solver3d.py      # 3D: the trace-degree effect

# then render it
python scripts/figure_4.py

# the result is written to
#   results/figs/fig04_consistency.png
```

A single figure can therefore be reproduced without running the whole suite, provided the
driver that writes its input record has been run once. Figure 4 takes about 5 seconds once
its records exist, and about 20 seconds from nothing.

### 3.3 Table-to-script map

| reported item | script |
|---|---|
| Table 1 — energy matching | `src/theory_energy_matching.py` |
| Table 2 — self-adjointness | `src/audit_dtn_adjoint.py` |
| Table 3 — 3D multi-load | `src/coupled_solver3d_mortar.py` |
| Table 4 — function-valued parameter | `src/func_operator.py` |
| Table 5 — learning curves | `src/func_operator.py` |
| Table 6 — interface-operator reconstruction | `src/deeponet_surrogate.py` |
| Table 7 — fair cost comparison | `src/cost_comparison.py` |
| Table 8 — budget comparison | `src/cost_report.py` |
| Sec. 5.3 — conditions for the law | `src/verify_p2_claim.py`, `src/chi_scalar.py` |
| Sec. 5.4 — strong-form null space | `src/strongform_nullspace.py`, `src/m6_energy_check.py` |
| Sec. 5.5 — independent verification | `src/patch_independent.py` |
| Sec. 5.6 — multiple interfaces | `src/multi_interface.py`, `src/multi_iface_1d.py` |
| Sec. 5.7 — 3D fields | `src/coupled_solver3d_mortar.py` |
| Sec. 5.8 — stress recovery | `src/stress_recovery2d.py`, `src/stress_recovery3d.py` |
| Sec. 5.9 — parameter type | `src/func_operator.py`, `src/param_operator.py` |
| Sec. 5.10 — strong-form comparison | `src/ddpinn_run.py`, `src/ddpinn_func.py` |
| Sec. 5.11 — curved interface, contrast | `src/annulus_benchmark.py` |
| Sec. 5.12 — seed statistics | `src/train_stats_cost.py` |
| Sec. 5.13 — cost accounting | `src/cost_report.py` |
| Sec. 5.14 — direct assembly vs iterative | `src/m2_iterative.py`, `src/adaptive_interface.py` |
| Lame cylinder (negative result) | `src/lame_cylinder.py`, `src/lame_referee.py` |

Every script prints its own tables and writes a plain-text record into `results/`. The
scripts in `scripts/` create that directory on first use, so it is not stored in the
repository; a driver from `src/` is run once to write the record a figure reads.

---

## 4. Data

The repository carries the **inputs the figures need**, not a large dataset. The
simulation domain is a rectangular plate under four load cases; there are no human
subjects, no personal data, and no ethical restrictions, so no anonymisation is required.

**Recorded arrays (`data/`, 1.1 MB):**

| file | what it holds |
|---|---|
| `deeponet_params.npz` | trained DeepONet weights for Table 6 |
| `ddpinn_budget_data.npz` | baseline budget results for Section 5.10 |
| `ddpinn_func_data.npz` | baseline accuracy records |
| `func_data_tr160_35.npz`, `func_data_tr192_60.npz` | training sets, two resolutions |
| `func_data_te32_60.npz` | held-out test fields |
| `func_data_ex20_75.npz` | extrapolation fields (contrast 20, 75) |

**Text records read by the figures:**

| file | read by |
|---|---|
| `theory_energy_matching.txt` | Figure 2 (the matching sweep) |
| `func_operator.txt` | Figure 8 and the architecture panel |
| `audit_dtn3d.txt` | the 3D comparison panel |

**Not included:** the full finite-element solutions (several GB). They are regenerated by
`src/coupled_solver3d.py` and `src/coupled_solver3d_mortar.py` in minutes; the archived
release on Zenodo carries the large fields for anyone who needs them without re-running.

**Path convention:** all paths are relative to the repository root, and no script contains
an absolute one. Two environment variables relocate the working directories:

```bash
RESULTS_DIR=/scratch/results python scripts/figure_1.py   # where outputs are written
DATA_DIR=/scratch/inputs    python scripts/figure_2.py    # where inputs are read
```

---

## 5. Random seeds and reproducibility

Every stochastic step is seeded explicitly. The manuscript reports **ensemble means with
standard deviations**, never single-seed values, and the seeds below are the ones used.

| where | seeds | file |
|---|---|---|
| two-invariant sweep, matching law (Figs. 1–2) | `3`, `7` | `scripts/figs_paper.py` |
| surrogate training, function-valued parameter | `0`, `1` | `src/func_operator.py` |
| interface-operator learning | `0` | `src/param_operator.py` |
| energy-driven coupled solver | `0`, `1` | `src/hybrid_energy_nn.py` |
| operator-learning variant | `0`, `11` | `src/hybrid_operator.py` |
| penalised variant | `7` | `src/hybrid_strategyA.py` |
| nonlinear coupled solver | `0` | `src/nonlinear_2d_final.py` |
| five-seed ensemble (Sec. 5.12) | `0`–`4` | `src/train_stats_cost.py` |

Exact reproduction of the learning results additionally requires the recorded arrays in
`data/`; the deep-learning runs are non-deterministic across hardware, so the reported
accuracy is an ensemble property rather than a single-run value.

**One caveat, stated plainly.** `scripts/fig06_upgrade.py` and `scripts/fig08_upgrade.py`
contain a small number of published constants (`LIN`, `DN_ENS`, `DN_M`, `DN_S` in the
latter) rather than reading them from a record. Figures 6 and 8 therefore **render** the
reported values but do not **recompute** them; the values themselves come from
`src/coupled_solver3d_mortar.py`, `src/func_operator.py` and `src/train_stats_cost.py`.
Figures 1–5, 7 and 9 are computed from records with no constants of this kind.

---

## 6. What the code does not establish

The manuscript states the limits of its own claims; they are repeated here so the code is
not read as proving more than it does.

* The energy-matching law is **proved for the series idealisation** and verified
  numerically in configurations that satisfy its stated conditions. The conditions
  (`(P1)` series additivity, `(P3)` one-sided drive) are listed in the manuscript, and the
  configurations that fall outside them are tested as well — including the
  traction-free-far-end case, whose earlier claim was **withdrawn**.
* The consistency proposition establishes **consistency** of the two transfer pairings.
  **No inf-sup condition and no a priori error bound are established.** The convergence
  rates the code reports are **measured** by a three-grid Richardson procedure, not
  derived from an estimate. In three dimensions no rate is reported, because the error
  there is governed by the trace degree rather than by the mesh size.

---

## 7. Repository layout

```
.
├── README.md                 this file
├── requirements.txt          dependencies, with the tested versions
├── LICENSE                   MIT
├── CITATION.cff              citation metadata (paper DOI added on acceptance)
├── .zenodo.json              archival metadata
│
├── scripts/                  one entry point per figure, plus the figure library
│   ├── figure_1.py  …  figure_9.py
│   ├── _paths.py             puts src/ on the import path
│   ├── figstyle.py           shared style: sizes, fonts, colour maps
│   ├── figs_paper.py         drawing code for Figures 1–4
│   ├── figs_fields.py        drawing code for Figures 5–9
│   ├── fig06_upgrade.py, fig06_redesign.py, fig08_upgrade.py
│   ├── ddpinn_fig.py         the baseline comparison panel
│   └── audit_dtn3d.py, audit_dtn_adjoint.py
│
├── src/                      the solvers
│   ├── coupled_solver*.py    interface assembly, the two transfer pairings
│   ├── theory_energy_matching.py, verify_p2_claim.py, chi_*.py
│   ├── multi_interface.py, multi_iface_1d.py, m2_iterative.py
│   ├── stress_recovery*.py, patch_independent.py, strongform_nullspace.py
│   ├── nonlinear_*.py        the total-Lagrangian extension
│   ├── func_operator.py, param_operator.py, hybrid_*.py, deeponet_*.py, ddpinn_*.py
│   ├── cost_report.py, cost_comparison.py, train_stats_cost.py
│   └── lame_*.py, annulus_benchmark.py
│
├── data/                     the inputs the figures need: recorded arrays and records
└── submission/               REPRODUCE.md — the per-figure reproduction map

results/ is not stored; it is created by the scripts when they run.
```

---

## 8. License and citation

Released under the **MIT License** — see `LICENSE`.

If you use this code, please cite the manuscript. `CITATION.cff` carries the metadata;
GitHub renders a **"Cite this repository"** button from it. The DOI of the archived
release is recorded in the Data availability statement of the manuscript and will be added
to `CITATION.cff` and `.zenodo.json` once the deposit is made.
