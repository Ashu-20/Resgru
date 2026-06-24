# ResGRU Decoder

A recurrent neural network decoder for the **heavy-hexagon quantum error correcting code**, built with [Stim](https://github.com/quantumlib/Stim) and [PyTorch](https://pytorch.org/).

The decoder uses a GRU encoder with a residual MLP head (`Branched_GRU`) to classify logical errors from syndrome measurement sequences, and supports multi-GPU distributed training via PyTorch DDP.

---

## Features

- Heavy-hexagon circuit generation for X and Z basis memory experiments (distance 3–9+)
- On-the-fly Stim simulation during training (no pre-generated data required)
- GRU decoder with residual MLP head and early stopping
- Multi-round training: single model trained across variable syndrome sequence lengths
- Multi-GPU training via `torchrun` / PyTorch DDP
- Tested on AMD MI250X GPUs (ROCm/HIP) on the LUMI supercomputer

---

## Installation

```bash
git clone https://github.com/ashutosh-kumar/resgru-decoder.git
cd resgru-decoder
pip install -e .
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Stim ≥ 1.13
- NumPy ≥ 1.24

See `requirements.txt` for the full pinned list.

---

## Repository Structure

```
resgru-decoder/
├── hhd/                        # Core installable package
│   ├── circuit/
│   │   └── hh_circ.py          # Heavy-hex Stim circuit generator
│   ├── models/
│   │   └── gru_decoder.py      # Branched_GRU model + ResBlock1D
│   └── data/
│       └── datasets.py         # HHLazyDataset, HHStimDataset, collate_fn
├── scripts/
│   ├── generate_data.py        # Pre-generate .npy syndrome datasets
│   └── train.py                # Main DDP training entry point
├── configs/
│   └── d5_example.yaml         # Example config for distance-5 training
└── docs/
    └── circuit_layout.md       # Heavy-hex qubit layout and detector explanation
```

---

## Quick Start

### 1. Generate syndrome data (optional — skip if using on-the-fly Stim)

```bash
python scripts/generate_data.py \
    --distance 5 \
    --rounds_list 3 5 7 9 11 13 15 \
    --num_shots 10000000 \
    --noise_max 1e-3 \
    --basis Z \
    --outdir /path/to/data
```

### 2. Train the decoder

**Single GPU:**
```bash
python scripts/train.py \
    --data_dir /path/to/data \
    --distance 5 \
    --rounds 3 5 7 9 11 13 15 \
    --hidden_size 256 \
    --num_layers 2 \
    --epochs 100 \
    --train_mode stim \
    --stim_noise 1e-3 \
    --stim_train_samples 1000000
```

**Multi-GPU (e.g. 4 GPUs):**
```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --data_dir /path/to/data \
    --distance 5 \
    --rounds 3 5 7 9 11 13 15 \
    --hidden_size 256 \
    --num_layers 2 \
    --epochs 100 \
    --train_mode stim \
    --stim_noise 1e-3 \
    --stim_train_samples 10000000
```

---

## On-the-fly vs Pre-generated Data

| Mode | `--train_mode` | Description |
|---|---|---|
| On-the-fly | `stim` | Circuits compiled once per process; Stim samples batches during training. No disk I/O bottleneck. |
| Pre-generated | `npy` | Loads `.npy` syndrome arrays from disk. Useful for fixed dataset comparisons. |

Val and test sets always use pre-generated `.npy` files for reproducibility.

---

## Noise Model

All four depolarizing noise parameters are set to the same physical error rate `p` by default:

| Parameter | Description |
|---|---|
| `after_clifford_depolarization` | Depolarization after each 1Q/2Q gate |
| `before_round_data_depolarization` | Depolarization on data qubits at round start |
| `before_measure_flip_probability` | Bit-flip before measurement |
| `after_reset_flip_probability` | Bit-flip after reset |

The `*(2/3)` scaling applied internally converts from a parameterized depolarizing convention where `p` represents the total error probability.

---

## HPC / LUMI Notes

For running on LUMI (AMD MI250X, ROCm):

- Use `torchrun` inside a Singularity container
- Disable MIOpen find-db cache to avoid SQLite contention across DDP ranks:
  ```bash
  export MIOPEN_FIND_MODE=FAST
  export MIOPEN_DEBUG_DISABLE_FIND_DB=1
  export MIOPEN_DISABLE_CACHE=1
  ```
- Stage large `.npy` files to `/tmp` (tmpfs) before training to avoid Lustre I/O bottlenecks
- Set `OMP_NUM_THREADS` = total CPUs / number of GPUs

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kumar2024resgru,
  author = {Kumar, Ashutosh},
  title  = {ResGRU Decoder: A Recurrent Neural Network Decoder for the Heavy-Hexagon Code},
  year   = {2024},
  url    = {https://github.com/ashutosh-kumar/resgru-decoder}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
