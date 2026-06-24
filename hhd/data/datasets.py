"""
Dataset classes for heavy-hexagon syndrome data.

HHLazyDataset
    Map-style dataset backed by pre-generated ``.npy`` syndrome files.
    Supports multiple rounds in a single dataset by concatenating across
    round-indexed file pairs.  Uses an LRU-capped mmap cache to avoid
    unbounded memory growth with many workers.

HHStimDataset
    Iterable-style dataset that generates syndrome samples on-the-fly
    using Stim.  Avoids disk I/O entirely; circuits are compiled once
    per process and reused across items.

collate_fn
    Pads variable-length syndrome sequences and stacks all fields into
    batch tensors suitable for Branched_GRU.

load_hh_packages
    Helper that validates and returns package metadata for HHLazyDataset.
"""

import os
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from hhd.circuit.hh_circ import generate_circuit


# ---------------------------------------------------------------------------
# Disk-backed dataset (pre-generated .npy files)
# ---------------------------------------------------------------------------

def load_hh_packages(
    data_dir: str,
    num_rounds: List[int],
    distance: int,
) -> List[dict]:
    """Return lightweight package metadata for each round.

    Each package dict contains:
    ``synd_path``, ``lbl_path``, ``r`` (rounds), ``n`` (number of shots).

    Parameters
    ----------
    data_dir : str
        Directory containing files named ``detection_d{d}_r{r}.npy`` and
        ``observable_d{d}_r{r}.npy``.
    num_rounds : list of int
        Round values to include.
    distance : int
        Code distance.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing.
    """
    pkgs = []
    for r in num_rounds:
        synd_path = os.path.join(data_dir, f"detection_d{distance}_r{r}.npy")
        lbl_path = os.path.join(data_dir, f"observable_d{distance}_r{r}.npy")
        if not (os.path.exists(synd_path) and os.path.exists(lbl_path)):
            raise FileNotFoundError(
                f"Missing .npy files for r={r} in {data_dir}. "
                f"Expected: detection_d{distance}_r{r}.npy and observable_d{distance}_r{r}.npy"
            )
        synd = np.load(synd_path, mmap_mode="r")
        lbl = np.load(lbl_path, mmap_mode="r")
        assert len(synd) == len(lbl), (
            f"Length mismatch for r={r}: {len(synd)} syndromes vs {len(lbl)} labels"
        )
        pkgs.append(
            {"synd_path": synd_path, "lbl_path": lbl_path, "r": r, "n": len(synd)}
        )
    return pkgs


class HHLazyDataset(torch.utils.data.Dataset):
    """Map-style dataset backed by memory-mapped ``.npy`` syndrome files.

    Samples are drawn uniformly across all rounds.  The underlying ``.npy``
    arrays are opened lazily and kept in an LRU cache (``max_open`` entries)
    to bound memory usage when used with many DataLoader workers.

    Parameters
    ----------
    packages : list of dict
        Output of :func:`load_hh_packages`.
    distance : int
        Code distance; determines the syndrome slice sizes.
    max_open : int
        Maximum number of simultaneously open mmap file pairs (default 4).
    """

    def __init__(self, packages: List[dict], distance: int, max_open: int = 4):
        self.packages = packages
        self.distance = distance
        self.max_open = max_open
        self.cum = np.cumsum([0] + [p["n"] for p in packages])

        # Derived dimensions
        self.dpr = int(((distance - 1) * (distance + 3)) / 2)
        self.num_z = int((distance - 1) + ((distance - 1) ** 2) / 2)

        # LRU cache: OrderedDict keyed by (synd_path, lbl_path)
        self._memo: OrderedDict = OrderedDict()

    def __len__(self) -> int:
        return int(self.cum[-1])

    def _open(self, p: dict):
        """Return (synd_array, lbl_array) from LRU cache, opening if needed."""
        key = (p["synd_path"], p["lbl_path"])
        if key in self._memo:
            self._memo.move_to_end(key)
        else:
            if len(self._memo) >= self.max_open:
                self._memo.popitem(last=False)  # evict least recently used
            self._memo[key] = (
                np.load(p["synd_path"], mmap_mode="r"),
                np.load(p["lbl_path"], mmap_mode="r"),
            )
        return self._memo[key]

    def __getitem__(self, idx: int):
        pkg_idx = int(np.searchsorted(self.cum, idx, side="right") - 1)
        p = self.packages[pkg_idx]
        i = idx - self.cum[pkg_idx]
        synd, lbl = self._open(p)

        row = synd[i]  # (N,)
        r = p["r"]

        init = row[: self.num_z]
        core = row[self.num_z : -self.num_z]
        final = row[-self.num_z :]

        main = core.reshape(r - 1, self.dpr)
        seq = torch.from_numpy(np.asarray(main, dtype=np.float32))
        final_t = torch.from_numpy(np.asarray(final, dtype=np.float32))
        init_t = torch.from_numpy(np.asarray(init, dtype=np.float32))

        li = lbl[i]
        label = torch.tensor(float(li.item() if hasattr(li, "item") else li), dtype=torch.float32)
        return seq, final_t, init_t, label, seq.shape[0]


# ---------------------------------------------------------------------------
# On-the-fly Stim dataset
# ---------------------------------------------------------------------------

class HHStimDataset(torch.utils.data.Dataset):
    """On-the-fly Stim dataset — no pre-generated data required.

    Each ``__getitem__`` call:
    1. Picks a random ``rounds`` value from ``rounds``.
    2. Uses a lazily compiled :class:`stim.CompiledDetectorSampler` for that
       ``rounds`` value (one per process, cached after first use).
    3. Samples one shot of detection events and the logical observable.
    4. Returns ``(seq, final_det, init_det, label, length)`` matching the
       layout of :class:`HHLazyDataset`.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : list of int
        Round values to sample from uniformly at random.
    samples_per_epoch : int
        Virtual dataset length (controls steps per epoch).
    noise_p : float
        Physical error probability passed to all four noise parameters.
    seed : int
        Base RNG seed; per-sample seed is ``seed + idx``.
    memory_basis : {"X", "Z"}
        Memory experiment basis (default "Z").
    """

    def __init__(
        self,
        distance: int,
        rounds: List[int],
        samples_per_epoch: int,
        noise_p: float,
        seed: int = 0,
        memory_basis: str = "Z",
    ):
        super().__init__()
        self.distance = distance
        self.rounds = rounds
        self.samples_per_epoch = samples_per_epoch
        self.noise_p = noise_p
        self.base_seed = seed
        self.memory_basis = memory_basis.upper()

        self.dpr = int(((distance - 1) * (distance + 3)) / 2)
        self.num_z = int((distance - 1) + ((distance - 1) ** 2) / 2)

        self._samplers: dict = {}  # r -> stim.CompiledDetectorSampler

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _get_sampler(self, r: int):
        """Lazily build and compile the Stim circuit for *r* rounds."""
        if r not in self._samplers:
            circ = generate_circuit(
                rounds=r,
                distance=self.distance,
                memory_basis=self.memory_basis,
                after_clifford_depolarization=self.noise_p,
                after_reset_flip_probability=self.noise_p,
                before_measure_flip_probability=self.noise_p,
                before_round_data_depolarization=self.noise_p,
            )
            self._samplers[r] = circ.compile_detector_sampler()
        return self._samplers[r]

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.base_seed + idx)
        r = int(rng.choice(self.rounds))
        sampler = self._get_sampler(r)

        dets, obs = sampler.sample(shots=1, separate_observables=True)
        det_row = np.asarray(dets[0], dtype=np.float32)

        init = det_row[: self.num_z]
        core = det_row[self.num_z : -self.num_z]
        final = det_row[-self.num_z :]

        main = core.reshape(r - 1, self.dpr).astype(np.float32)
        label_val = float(obs[0, 0])

        seq = torch.from_numpy(main)
        final_t = torch.from_numpy(final)
        init_t = torch.from_numpy(init)
        label_t = torch.tensor(label_val, dtype=torch.float32)

        return seq, final_t, init_t, label_t, seq.shape[0]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Collate a list of ``(seq, final, init, label, length)`` tuples.

    Pads ``seq`` to the longest sequence in the batch.

    Returns
    -------
    tuple
        ``(padded_seq, finals, initials, labels, lengths)`` where shapes are
        ``(B, T_max, C)``, ``(B, z_stab)``, ``(B, z_stab)``, ``(B,)``, ``(B,)``.
    """
    seq, fm, idet, label, length = zip(*batch)
    padded_seq = nn.utils.rnn.pad_sequence(seq, batch_first=True)  # (B, T, C)
    finals = torch.stack(fm)       # (B, z_stab)
    initials = torch.stack(idet)   # (B, z_stab)
    labels = torch.stack(label)    # (B,)
    lengths = torch.tensor(length, dtype=torch.long)  # (B,)
    return padded_seq, finals, initials, labels, lengths
