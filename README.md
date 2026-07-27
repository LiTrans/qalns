# qalns

Reference implementation of the DQN-guided quantum–classical Adaptive Large Neighborhood Search (ALNS) framework for the Pickup-and-Delivery Problem with Time Windows (PDPTW), described in

> F. Moosavi and B. Farooq, *RL-Guided Quantum-ALNS for Constrained Vehicle Routing*, IEEE International Conference on Quantum Computing and Engineering (QCE), 2026.

Shallow gate-based quantum samplers (QAOA and EfficientSU2) are embedded inside the repair phase of an ALNS heuristic. A Double-DQN controller selects, at each iteration, between classical repair operators and a quantum sampler, using entropy, feasibility, and hardware-noise features.

## Layout

```
qalns/
├── src/                              Python modules
│   ├── pdptw.py                      PDPTW data model, evaluator, instance generator
│   ├── alns.py                       ALNS destroy–repair loop
│   ├── repair.py                     Destroy/repair operators (classical + Qiskit)
│   ├── entropy.py                    Entropy features
│   ├── policy.py                     Entropy-aware repair policy
│   ├── hardware_noise.py             Empirical noise-aware predictor
│   ├── dqn_policy.py                 Double-DQN Q-network and policy
│   ├── dqn_replay_buffer.py          Offline replay buffer
│   ├── dqn_train.py                  Double-DQN training loop
│   ├── rl_training.py                Contextual-bandit data collection and training
│   └── experiment_runner.py          End-to-end experiment driver
├── data/                             CSV artifacts used in the paper
│   ├── hardware_runs.csv             Raw ibm_quebec calibration log
│   ├── benchmark_grid_summary.csv    Li–Lim benchmark grid, all methods, all seeds
│   ├── grid_alns_comparison_summary_15.csv    Fixed-budget grid, (tw=0.15, cap=0.15)
│   └── grid_alns_comparison_summary_85.csv    Fixed-budget grid, (tw=0.85, cap=0.85)
├── models/
│   └── hardware_noise_model_ibm_quebec.json   Pre-trained noise-aware predictor
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or later
- NumPy, Matplotlib, Qiskit, Qiskit Aer, Qiskit IBM Runtime, rustworkx

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install qiskit
pip install qiskit_ibm_runtime
pip install rustworkx
pip install 'qiskit[visualization]'
pip install qiskit_aer
```

Qiskit is required both for the QAOA / EfficientSU2 samplers and for the transpilation used when calibrating the empirical noise-aware predictor.

## Usage

### 1. Train the empirical noise-aware predictor

Trains the calibrated predictor from the raw hardware log.

```bash
python src/hardware_noise.py train \
    --hardware-csv data/hardware_runs.csv \
    --model-path models/hardware_noise_model_ibm_quebec.json
```

The distributed `hardware_noise_model_ibm_quebec.json` was produced by this command on 960 matched hardware/Aer pairs.

### 2. Collect offline contextual-bandit data

```bash
python src/rl_training.py collect \
    --out-csv runs/training_actions.csv \
    --sizes 15,20 \
    --seeds 1,2,3,4,5 \
    --iterations 100 \
    --remove-counts 2,3,4,5 \
    --candidate-caps 2,3,4 \
    --quantum-eval-ratio 0.10 \
    --hardware-noise-model models/hardware_noise_model_ibm_quebec.json
```

Full-scale collection takes several hours on a laptop; reduce `--sizes`, `--seeds`, or `--iterations` for a quick sanity run.

### 3. Train the ridge contextual bandit

```bash
python src/rl_training.py train \
    --data-csv runs/training_actions.csv \
    --model-path runs/rl_policy.json
```

### 4. Train the Double-DQN

```bash
python src/dqn_train.py split-dataset \
    --data-csv runs/training_actions.csv \
    --train-csv runs/train_actions.csv \
    --test-csv runs/test_actions.csv

python src/dqn_train.py train-dataset \
    --data-csv runs/train_actions.csv \
    --model-path runs/dqn_policy.json \
    --replay-path runs/dataset_replay.json
```

### 5. Run the fixed-budget ALNS grid

```bash
python src/experiment_runner.py \
    --out-dir runs/ \
    --policy-model runs/rl_policy.json \
    --hardware-noise-model models/hardware_noise_model_ibm_quebec.json \
    --sizes 100,150,200 \
    --seeds 1,2,3,4,5 \
    --iterations 150 \
    --remove-counts 2,3,4,5 \
    --candidate-caps 2,3,4 \
    --qiskit-max-states 1024 \
    --quantum-eval-ratio 0.25
```

Add `--use-qiskit-aer` for real Qiskit Aer circuit execution (much slower).

## Default hyperparameters

Reported in Section IV.A of the paper.

- Double-DQN: 2-layer MLP, hidden 64 (ReLU); discount 0.99; batch 64; replay buffer 300 k; target hard-update every 50 steps; learning rate 1e-3; 1000 gradient steps.
- QAOA: `p ∈ {1, 2}`, standard X-mixer, cost Hamiltonian built via fast Walsh–Hadamard transform, initialization bank of 5 schemes.
- EfficientSU2: `L ∈ {1, 2}`, `{Rx, Ry, Rz}` rotations, linear entanglement, initialization bank of 5 schemes.
- Transpiler optimization level 3; target backend `ibm_quebec`.
- Shot budgets `{16, 128, 1024}`.

## Citation

```bibtex
@inproceedings{moosavi2026rlquantum,
  title     = {{RL}-Guided Quantum-{ALNS} for Constrained Vehicle Routing},
  author    = {Moosavi, Farzan and Farooq, Bilal},
  booktitle = {IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year      = {2026}
}
```
