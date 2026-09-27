# OU-dRVFL-RL

**Uncertainty-Aware Deep Random Feature Reinforcement Learning with Online Recursive Least Squares Critic Adaptation**

Urvashi Sharma, Chirag Patel*
Department of Computer Engineering, Devang Patel Institute of Advance Technology and Research, CHARUSAT University

---

## Overview

`OU-dRVFL-RL` is a hybrid actor–critic reinforcement-learning framework whose **critic output weights are updated every transition by regularized recursive least squares (RLS)** over a **fixed deep random vector functional link (dRVFL)** feature map. An ensemble of independently randomized heads supplies a cheap epistemic-uncertainty proxy used for either upper-confidence exploration or temporally coherent randomized value-function exploration. A lightweight gradient-trained actor handles continuous actions.

The method is evaluated on classic control, MuJoCo continuous control, controlled distribution shift, uncertainty diagnostics, and a financial time-series case study. It is **not** a fully gradient-free learner: only the critic readout is RLS-updated.

## Key Contributions

- Per-transition regularized RLS critic over fixed deep random features with direct input–output links.
- Two uncertainty-aware exploration modes: UCB bonus and episode-wise randomized value functions.
- Exact RLS–batch equivalence, finite-sample convergence under explicit stationary assumptions, finite-time covariance growth bounds, and ensemble-variance / disagreement decompositions.
- Full, block-diagonal, and diagonal RLS complexity analysis with explicit RLS-vs-gradient cost conditions.
- Measured return, adaptation, uncertainty quality, latency, memory, energy, and numerical stability on NVIDIA Jetson Orin Nano.

## Headline Results (as reported in the paper)

| Task | SAC | TD3 | OU-dRVFL-RL (full) |
|---|---|---|---|
| Hopper (1M steps) | 2,691 ± 407 | 2,534 ± 389 | **2,847 ± 312** |
| Walker2d | 3,847 ± 512 | 3,612 ± 478 | **3,924 ± 445** |
| HalfCheetah | **9,103 ± 487** | 8,847 ± 521 | 8,912 ± 524 |
| Ant | **4,212 ± 689** | 3,987 ± 712 | <1,000 (failed) |

Jetson Orin Nano mean update latency: **0.9 ms** (diagonal RLS), **2.8 ms** (full RLS), **14.2 ms** (SAC).

On Ant, OU-dRVFL-RL failed for all seeds; this is documented as a fundamental limitation of the fixed feature map for high-dimensional torque control.

## Repository Layout

```
paper/        LaTeX source of the manuscript
src/          Core implementation (features, RLS critic, ensemble, exploration, actor, agent)
configs/      YAML configuration files for every reported experiment
scripts/      Training, evaluation, table/figure generation, power measurement, reference audit
results/      Seed-level CSVs, learning curves, power traces, generated tables
docs/         Environment, dataset, and hardware cards; reproducibility checklist; proofs
tests/        Unit tests for the mathematical claims (RLS equivalence, covariance bound, variance decomposition)
```

## Installation

```bash
git clone https://github.com/chirag453/OU-dRVFL-RL.git
cd OU-dRVFL-RL
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` (recommended):

```
numpy>=1.26
scipy>=1.11
torch>=2.1
gymnasium==0.29.1
mujoco==3.1.4
pyyaml>=6.0
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.66
pytest>=8.0
```

## Reproducing the Paper

### 1. Train

```bash
python scripts/train.py --config configs/hopper_full.yaml --seed 0
```

Repeat for seeds `0..9`. Every reported number in the paper uses seeds `0..9`; seed `100` is reserved for hyperparameter search.

### 2. Evaluate

```bash
python scripts/evaluate.py --run results/seeds/hopper_full_seed0 --episodes 100
```

### 3. Generate Tables and Figures

```bash
python scripts/make_tables.py  --results results/seeds --out results/tables
python scripts/make_figures.py --results results/seeds --out results/curves
```

### 4. Measure Edge Power / Latency (Jetson Orin Nano)

```bash
python scripts/measure_power.py --config configs/hopper_full.yaml --n-updates 100000 --sessions 5
```

Each session includes thermal stabilization, an idle block, a workload block, and a post-workload idle block. Energy per update uses trapezoidal integration with idle correction (see `docs/hardware_cards.md`).

### 5. Build the Paper

```bash
make paper
```

## Configuration Format

Every experiment is fully specified in YAML, e.g. `configs/hopper_full.yaml`:

```yaml
env: Hopper-v4
seed: 0
feature:
  depth: 2
  width: 256
  activation: relu
  direct_link: true
critic:
  ensemble_size: 5
  covariance: full        # full | block | diagonal
  forgetting: 0.995
  ridge: 0.001
  bootstrap_prob: 0.8
  prior_scale: 0.3
  innovation_clip: 10.0
  denom_floor: 1.0e-8
exploration:
  mode: ucb               # ucb | randomized_value
  beta: 0.1
actor:
  hidden_width: 256
  lr: 3.0e-4
  target_tau: 0.005
  policy_delay: 2
training:
  total_steps: 1000000
  replay_capacity: 1000000
  warmup_steps: 25000
  batch_size: 256
```

## Mathematical Guarantees (with Scope Tags)

| Result | Tag | Scope |
|---|---|---|
| Theorem 1: RLS–batch equivalence | **Exact identity** | Any sequence; no stationarity needed |
| Theorem 2: Finite-sample convergence | **Stationary only** | Fixed target, martingale noise, uniform PE |
| Theorem 3: Covariance growth bound | **Exact identity** | Finite-time only; not uniform stability |
| Theorem 4: Ensemble variance decomposition | **Exact identity** | Any jointly square-integrable heads |
| Theorem 5: Expected disagreement decomposition | **Heuristic extension** | Conditional independence; idealized |
| Theorem 6: RLS complexity | **Exact identity** | Full / block / diagonal |

Proofs and scope remarks are in the manuscript (Section 3) and condensed in `docs/mathematical_proofs.md`.

## Tests

```bash
pytest tests/ -v
```

- `test_rls_equivalence.py` — numerically checks Theorem 1 against a batch ridge solve.
- `test_covariance_bounds.py` — checks the finite-time bound of Theorem 3.
- `test_ensemble_variance.py` — checks Theorem 4 by Monte Carlo.

## Reproducibility Package

- **Code**: this repository
- **Paper**: `paper/verifiedresylt27sep-3.42pm.tex`
- **Raw results**: `results/seeds/` (seed-level CSVs), `results/curves/` (learning curves), `results/power/` (Jetson power traces)
- **Permanent DOI**: via Zenodo (see `CITATION.cff` and `codemeta.json`)
- **Independent replication**: reported in the manuscript

## Citation

If you use this code, please cite:

```bibtex
@article{sharma_patel_oudrvfl_2025,
  title   = {Uncertainty Aware Deep Random Feature Reinforcement Learning with Online Recursive Least Squares Critic Adaptation},
  author  = {Sharma, Urvashi and Patel, Chirag},
  journal = {Manuscript},
  year    = {2025},
  note    = {Code: https://github.com/chirag453/OU-dRVFL-RL}
}
```

See `CITATION.cff` for a machine-readable version.

## License

Released under the MIT License. See `LICENSE`.

## Contact

- Chirag Patel — chiragpatel.dce@charusat.ac.in
- Urvashi Sharma — urvashichaudhari.dce@charusat.ac.in