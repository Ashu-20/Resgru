"""
Heavy-hexagon Stim circuit generator.

Supports X and Z basis memory experiments for arbitrary code distance d >= 3
(odd distances recommended for the heavy-hex layout).

Public API
----------
generate_circuit(*, rounds, distance, memory_basis, **noise_kwargs) -> stim.Circuit
    Convenience wrapper matching the generate_circuit API used by training scripts.

generate_heavy_hex_circuit(params, *, memory_basis) -> stim.Circuit
    Full circuit builder accepting a Circuitgen parameter dataclass.

Noise model
-----------
Four independent depolarizing noise channels are supported:

  after_clifford_depolarization      -- after every 1Q / 2Q Clifford gate
  before_round_data_depolarization   -- on data qubits at the start of each round
  before_measure_flip_probability    -- bit-flip immediately before measurement
  after_reset_flip_probability       -- bit-flip immediately after reset

All four accept a physical error probability p in [0, 1).  Internally the
*(2/3) prefactor converts from the parameterisation where p is the total
error probability of the depolarizing channel to the per-Pauli probability
expected by Stim's DEPOLARIZE1/DEPOLARIZE2 instructions.
"""

import stim
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def append_anti_basis_error(
    circuit: stim.Circuit,
    targets: List[int],
    p: float,
    basis: str,
) -> None:
    """Apply the anti-basis single-qubit error with probability *p*.

    For an X-basis operation the anti-basis error is Z, and vice-versa.
    No instruction is emitted when p == 0.
    """
    if p > 0:
        if basis == "X":
            circuit.append_operation("Z_ERROR", targets, p)
        else:
            circuit.append_operation("X_ERROR", targets, p)


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class Circuitgen:
    """Circuit generation parameters.

    Attributes
    ----------
    rounds : int
        Number of syndrome extraction rounds (>= 1).
    d : int
        Code distance.  Must be an odd integer >= 3 for the heavy-hex layout.
    after_clifford_depolarization : float
        Depolarizing noise applied after every 1Q/2Q Clifford (default 0).
    before_round_data_depolarization : float
        Depolarizing noise on data qubits at the start of each round (default 0).
    before_measure_flip_probability : float
        Bit-flip probability applied before each measurement (default 0).
    after_reset_flip_probability : float
        Bit-flip probability applied after each reset (default 0).
    """

    rounds: int
    d: int = None
    after_clifford_depolarization: float = 0
    before_round_data_depolarization: float = 0
    before_measure_flip_probability: float = 0
    after_reset_flip_probability: float = 0

    def get_index_cord(self, cord: Tuple[int, int]) -> int:
        """Convert a 2-D grid coordinate to a flat qubit index."""
        return (2 * self.d - 1) * cord[0] + cord[1]

    # ------------------------------------------------------------------
    # Low-level circuit primitives
    # ------------------------------------------------------------------

    def append_begin_round_tick(
        self,
        circuit: stim.Circuit,
        data_qubits_ind: List[int],
    ) -> None:
        """Insert a TICK and optional data-qubit depolarization at round start."""
        circuit.append_operation("TICK", [])
        if self.before_round_data_depolarization > 0:
            circuit.append_operation(
                "DEPOLARIZE1",
                data_qubits_ind,
                self.before_round_data_depolarization * (2 / 3),
            )

    def append_measure(
        self,
        circuit: stim.Circuit,
        target: List[int],
        basis: str = "Z",
    ) -> None:
        """Measure *target* qubits in *basis* with optional pre-measurement noise."""
        append_anti_basis_error(
            circuit, target, self.before_measure_flip_probability * (2 / 3), basis
        )
        circuit.append_operation("M" + basis, target)

    def append_reset(
        self,
        circuit: stim.Circuit,
        targets: List[int],
        basis: str = "Z",
    ) -> None:
        """Reset *targets* in *basis* with optional post-reset noise."""
        circuit.append_operation("R" + basis, targets)
        append_anti_basis_error(
            circuit, targets, self.after_reset_flip_probability * (2 / 3), basis
        )

    def append_measure_reset(
        self,
        circuit: stim.Circuit,
        targets: List[int],
        basis: str = "Z",
    ) -> None:
        """Measure-and-reset *targets* in *basis* with pre/post noise."""
        append_anti_basis_error(
            circuit, targets, self.before_measure_flip_probability * (2 / 3), basis
        )
        circuit.append_operation("MR" + basis, targets)
        append_anti_basis_error(
            circuit, targets, self.after_reset_flip_probability * (2 / 3), basis
        )

    def append_unitary_1(
        self,
        circuit: stim.Circuit,
        name: str,
        targets: List[int],
    ) -> None:
        """Apply a single-qubit Clifford gate with optional depolarizing noise."""
        circuit.append_operation(name, targets)
        if self.after_clifford_depolarization > 0:
            circuit.append_operation(
                "DEPOLARIZE1", targets, self.after_clifford_depolarization
            )

    def append_unitary_2(
        self,
        circuit: stim.Circuit,
        name: str,
        targets: List[int],
    ) -> None:
        """Apply a two-qubit Clifford gate with optional depolarizing noise."""
        circuit.append_operation(name, targets)
        if self.after_clifford_depolarization > 0:
            circuit.append_operation(
                "DEPOLARIZE2", targets, self.after_clifford_depolarization
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rec_lookback_map(
    total_meas: int,
    ordered_coords: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], int]:
    """Map each ancilla coordinate to its measurement-record look-back offset.

    The *k*-th measurement in *ordered_coords* is addressed as
    ``rec[-(total_meas - k)]`` relative to the end of the current round's
    measurement record.
    """
    return {coord: total_meas - i for i, coord in enumerate(ordered_coords)}


# ---------------------------------------------------------------------------
# Main circuit builder
# ---------------------------------------------------------------------------

def generate_heavy_hex_circuit(
    params: Circuitgen,
    *,
    memory_basis: str = "Z",
) -> stim.Circuit:
    """Build a full heavy-hex memory-experiment Stim circuit.

    Parameters
    ----------
    params : Circuitgen
        Noise and structural parameters.
    memory_basis : {"X", "Z"}
        Logical basis for the memory experiment.

    Returns
    -------
    stim.Circuit
        Complete circuit including state preparation, syndrome rounds,
        final data-qubit measurement, detectors, and logical observable.
    """
    memory_basis = memory_basis.upper()
    if memory_basis not in {"X", "Z"}:
        raise ValueError("memory_basis must be 'X' or 'Z'")
    if params.rounds < 1:
        raise ValueError("Need rounds >= 1")
    if params.d is not None and params.d < 2:
        raise ValueError("Need a distance >= 2")

    d = params.d
    memory_in_x_basis = memory_basis == "X"

    # ------------------------------------------------------------------
    # Coordinate book-keeping
    # ------------------------------------------------------------------
    data_coords: List[Tuple[int, int]] = [
        (2 * i, 2 * j) for i in range(d) for j in range(d)
    ]
    z_gauge_coords: List[Tuple[int, int]] = [
        (2 * i, 2 * j + 1) for i in range(d) for j in range(d - 1)
    ]
    x_gauge_coords_bulk: List[Tuple[int, int]] = [
        (2 * i + 1, 2 * j + 1)
        for i in range(d - 1)
        for j in range(d - 1)
        if (i + j) % 2 == 0
    ]
    # Boundary (weight-2) X-gauges on the top/bottom rows
    x_gauge_coords_boundary: List[Tuple[int, int]] = [
        (4 * j + 3, 0) for j in range(int((d - 1) / 2))
    ] + [
        (4 * j + 1, 2 * d - 2) for j in range(int((d - 1) / 2))
    ]
    x_gauge_coords = x_gauge_coords_bulk + x_gauge_coords_boundary

    flag_coords: List[Tuple[int, int]] = [
        (2 * i, 2 * j + 1)
        for i in range(d - 1)
        for j in range(d - 1)
        if (i + j) % 2 == 0
    ] + [
        (2 * i + 2, 2 * j + 1)
        for i in range(d - 1)
        for j in range(d - 1)
        if (i + j) % 2 == 0
    ]

    # ------------------------------------------------------------------
    # Qubit index helpers
    # ------------------------------------------------------------------
    def qindex(coord: Tuple[int, int]) -> int:
        return (2 * d - 1) * coord[0] + coord[1]

    data_qubits = [qindex(c) for c in data_coords]
    z_gauge_qubits = [qindex(c) for c in z_gauge_coords]
    x_gauge_qubits = [qindex(c) for c in x_gauge_coords]
    flag_qubits = [qindex(c) for c in flag_coords]

    all_measure_coords = x_gauge_coords + flag_coords + z_gauge_coords
    all_measure_qubits = x_gauge_qubits + flag_qubits + z_gauge_qubits

    qubits_cord_seq = {
        tup: (2 * d - 1) * tup[0] + tup[1]
        for tup in (data_coords + x_gauge_coords + z_gauge_coords)
    }

    # ------------------------------------------------------------------
    # X-memory tail: data support for each vertical strip of X-gauges
    # ------------------------------------------------------------------
    def x_strip_data_coords(strip: int) -> List[Tuple[int, int]]:
        col = 2 * strip + 1
        coords_in_strip = [c for c in x_gauge_coords if c[0] == col]
        touched = set()
        for measure in coords_in_strip:
            candidates = [
                tuple(np.add(measure, (1, 1))),
                tuple(np.add(measure, (1, -1))),
                tuple(np.add(measure, (-1, 1))),
                tuple(np.add(measure, (-1, -1))),
                tuple(np.add(measure, (1, 0))),
                tuple(np.add(measure, (-1, 0))),
            ]
            for c in candidates:
                if c in data_coords:
                    touched.add(c)
        return sorted(touched)

    # ------------------------------------------------------------------
    # Single syndrome-extraction round
    # ------------------------------------------------------------------
    def _build_round() -> Tuple[stim.Circuit, stim.Circuit]:
        """Return (head_block, round_block).

        head_block  -- one round of gates *without* detectors (used for state-prep).
        round_block -- same gates *with* detectors and SHIFT_COORDS appended.
        """
        # ---- X-gauge CNOT schedule (5 time steps) ----
        cnot_target_x: List[List[int]] = [[], [], [], [], []]

        # Step 2: H on X gauges → entangle with upper flag
        for measure in x_gauge_coords_bulk:
            flag1 = tuple(np.add(measure, (1, 0)))
            cnot_target_x[0].append(int(qubits_cord_seq[measure]))
            cnot_target_x[0].append(int(qubits_cord_seq[flag1]))

        # Step 3
        for measure in x_gauge_coords_bulk:
            flag_1 = tuple(np.add(measure, (1, 0)))
            data1 = tuple(np.add(measure, (1, 1)))
            flag_2 = tuple(np.add(measure, (-1, 0)))
            if all(cord in qubits_cord_seq for cord in [flag_1, data1, flag_2]):
                cnot_target_x[1].extend([
                    int(qubits_cord_seq[flag_1]),
                    int(qubits_cord_seq[data1]),
                    int(qubits_cord_seq[measure]),
                    int(qubits_cord_seq[flag_2]),
                ])

        # Step 4
        for measure in x_gauge_coords_bulk:
            flag_1 = tuple(np.add(measure, (1, 0)))
            data1 = tuple(np.add(measure, (1, -1)))
            flag_2 = tuple(np.add(measure, (-1, 0)))
            data2 = tuple(np.add(measure, (-1, -1)))
            if all(coord in qubits_cord_seq for coord in [flag_1, data1, flag_2, data2]):
                cnot_target_x[2].extend([
                    int(qubits_cord_seq[flag_1]),
                    int(qubits_cord_seq[data1]),
                    int(qubits_cord_seq[flag_2]),
                    int(qubits_cord_seq[data2]),
                ])
        for measure in x_gauge_coords_boundary:
            if measure[1] == 2 * d - 2:
                data1 = tuple(np.add(measure, (1, 0)))
                cnot_target_x[2].append(int(qubits_cord_seq[measure]))
                cnot_target_x[2].append(int(qubits_cord_seq[data1]))

        # Step 5
        for measure in x_gauge_coords_bulk:
            flag_1 = tuple(np.add(measure, (1, 0)))
            data1 = tuple(np.add(measure, (-1, 1)))
            flag_2 = tuple(np.add(measure, (-1, 0)))
            if all(coord in qubits_cord_seq for coord in [flag_1, data1, flag_2]):
                cnot_target_x[3].extend([
                    int(qubits_cord_seq[measure]),
                    int(qubits_cord_seq[flag_1]),
                    int(qubits_cord_seq[flag_2]),
                    int(qubits_cord_seq[data1]),
                ])
        for measure in x_gauge_coords_boundary:
            if measure[1] == 2 * d - 2:
                data2 = tuple(np.add(measure, (-1, 0)))
                cnot_target_x[3].append(int(qubits_cord_seq[measure]))
                cnot_target_x[3].append(int(qubits_cord_seq[data2]))
        for measure in x_gauge_coords_boundary:
            if measure[1] == 0:
                data2 = tuple(np.add(measure, (1, 0)))
                cnot_target_x[3].append(int(qubits_cord_seq[measure]))
                cnot_target_x[3].append(int(qubits_cord_seq[data2]))

        # Step 6
        for measure in x_gauge_coords_bulk:
            flag_2 = tuple(np.add(measure, (-1, 0)))
            cnot_target_x[4].append(int(qubits_cord_seq[measure]))
            cnot_target_x[4].append(int(qubits_cord_seq[flag_2]))
        for measure in x_gauge_coords_boundary:
            if measure[1] == 0:
                data2 = tuple(np.add(measure, (-1, 0)))
                cnot_target_x[4].append(int(qubits_cord_seq[measure]))
                cnot_target_x[4].append(int(qubits_cord_seq[data2]))

        # ---- Z-gauge CNOT schedule (3 time steps) ----
        cnot_target_z: List[List[int]] = [[], [], []]
        for i in range(d):
            for j in range(d - 1):
                measure = (2 * i, 2 * j + 1)
                data_1 = (2 * i, 2 * j)
                data_2 = (2 * i, 2 * j + 2)
                if (i + j) % 2 == 0:
                    cnot_target_z[1].append(int(qindex(data_1)))
                    cnot_target_z[1].append(int(qindex(measure)))
                    cnot_target_z[2].append(int(qindex(data_2)))
                    cnot_target_z[2].append(int(qindex(measure)))
                else:
                    cnot_target_z[1].append(int(qindex(data_1)))
                    cnot_target_z[1].append(int(qindex(measure)))
                    cnot_target_z[0].append(int(qindex(data_2)))
                    cnot_target_z[0].append(int(qindex(measure)))

        # ---- Assemble gate sequence ----
        cycle_action = stim.Circuit()
        params.append_begin_round_tick(cycle_action, data_qubits)

        params.append_unitary_1(cycle_action, "H", x_gauge_qubits)
        for target in cnot_target_x:
            cycle_action.append_operation("TICK", [])
            params.append_unitary_2(cycle_action, "CNOT", target)
        cycle_action.append_operation("TICK", [])
        params.append_unitary_1(cycle_action, "H", x_gauge_qubits)
        cycle_action.append_operation("TICK", [])
        params.append_measure_reset(cycle_action, x_gauge_qubits)
        params.append_measure_reset(cycle_action, flag_qubits)

        cycle_action.append_operation("TICK", [])
        for target in cnot_target_z:
            params.append_unitary_2(cycle_action, "CNOT", target)
            cycle_action.append_operation("TICK", [])
        params.append_measure_reset(cycle_action, z_gauge_qubits)

        # Save the gate-only block (no detectors) for the head
        cycle_action_head = cycle_action.copy()

        # ---- Detector mapping ----
        total_meas = len(all_measure_qubits)
        rec_lookup = _build_rec_lookback_map(
            total_meas, x_gauge_coords + flag_coords + z_gauge_coords
        )
        rec_lookup_z = {c: rec_lookup[c] for c in z_gauge_coords}
        rec_lookup_x_gauge = {c: rec_lookup[c] for c in x_gauge_coords}

        # X-type Bacon-Shor stabilizers (each vertical strip)
        for strip in range(d - 1):
            col = 2 * strip + 1
            coords_in_strip = [c for c in x_gauge_coords if c[0] == col]
            if not coords_in_strip:
                continue
            cur_round = [stim.target_rec(-rec_lookup_x_gauge[c]) for c in coords_in_strip]
            prev_round = [
                stim.target_rec(-(rec_lookup_x_gauge[c] + total_meas))
                for c in coords_in_strip
            ]
            cycle_action.append_operation(
                "DETECTOR", cur_round + prev_round, [col, 2 * (d - 1), 0]
            )

        # Z-type stabilizers (weight-4 / weight-2 on boundaries)
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 != 0:
                    left = (2 * i, 2 * j + 1)
                    right = (2 * i + 2, 2 * j + 1)
                    if left in rec_lookup_z and right in rec_lookup_z:
                        targets = [
                            stim.target_rec(-rec_lookup_z[left]),
                            stim.target_rec(-rec_lookup_z[right]),
                            stim.target_rec(-(rec_lookup_z[left] + total_meas)),
                            stim.target_rec(-(rec_lookup_z[right] + total_meas)),
                        ]
                        cycle_action.append_operation(
                            "DETECTOR", targets, [2 * i + 1, 2 * j + 1, 0]
                        )

        for m in range(int((d - 1) / 2)):
            b_left = (0, 4 * m + 1)
            b_right = (2 * (d - 1), 4 * m + 3)
            for edge in [b_left, b_right]:
                if edge in rec_lookup_z:
                    cycle_action.append_operation(
                        "DETECTOR",
                        [
                            stim.target_rec(-rec_lookup_z[edge]),
                            stim.target_rec(-(rec_lookup_z[edge] + total_meas)),
                        ],
                        [edge[0], edge[1], 0],
                    )

        cycle_action.append_operation("SHIFT_COORDS", [], [0, 0, 1])
        return cycle_action_head, cycle_action

    # ------------------------------------------------------------------
    # Build head (state-prep + first syndrome round, no detectors)
    # ------------------------------------------------------------------
    head_block, round_block = _build_round()

    head = stim.Circuit()
    for coord in data_coords + x_gauge_coords + z_gauge_coords:
        head.append_operation(
            "QUBIT_COORDS", [qubits_cord_seq[coord]], [coord[0], coord[1]]
        )

    if memory_in_x_basis:
        params.append_reset(head, data_qubits, "X")
    else:
        params.append_reset(head, data_qubits, "Z")

    params.append_reset(head, x_gauge_qubits + z_gauge_qubits, "Z")

    rec_lookup = _build_rec_lookback_map(
        len(all_measure_coords), x_gauge_coords + flag_coords + z_gauge_coords
    )
    rec_lookup_z = {c: rec_lookup[c] for c in z_gauge_coords}
    rec_lookup_x_gauge = {c: rec_lookup[c] for c in x_gauge_coords}

    head += head_block

    # Head detectors (first round — always deterministic)
    if memory_in_x_basis:
        for strip in range(d - 1):
            col = 2 * strip + 1
            coords_in_strip = [c for c in x_gauge_coords if c[0] == col]
            if not coords_in_strip:
                continue
            targets = [stim.target_rec(-rec_lookup_x_gauge[c]) for c in coords_in_strip]
            head.append_operation("DETECTOR", targets, [col, 2 * (d - 1), 0])
    else:
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 != 0:
                    left = (2 * i, 2 * j + 1)
                    right = (2 * i + 2, 2 * j + 1)
                    if left in rec_lookup_z and right in rec_lookup_z:
                        targets = [
                            stim.target_rec(-rec_lookup_z[left]),
                            stim.target_rec(-rec_lookup_z[right]),
                        ]
                        head.append_operation(
                            "DETECTOR", targets, [2 * i + 1, 2 * j + 1, 0]
                        )
        for m in range(int((d - 1) / 2)):
            b_left = (0, 4 * m + 1)
            b_right = (2 * (d - 1), 4 * m + 3)
            for edge in [b_left, b_right]:
                if edge in rec_lookup_z:
                    head.append_operation(
                        "DETECTOR",
                        [stim.target_rec(-rec_lookup_z[edge])],
                        [edge[0], edge[1], 0],
                    )

    head.append_operation("SHIFT_COORDS", [], [0, 0, 1])

    # ------------------------------------------------------------------
    # Body: repeated syndrome rounds
    # ------------------------------------------------------------------
    body = stim.Circuit()
    body += round_block * (params.rounds - 1)

    # ------------------------------------------------------------------
    # Tail: final data-qubit measurement + detectors + logical observable
    # ------------------------------------------------------------------
    tail = stim.Circuit()
    final_measure_basis = "X" if memory_in_x_basis else "Z"
    params.append_measure(tail, data_qubits, final_measure_basis)

    data_rec_map = _build_rec_lookback_map(len(data_qubits), data_coords)

    # Retrieve the per-round record-lookup used in the tail
    total_meas = len(all_measure_qubits)
    rec_lookup_tail = _build_rec_lookback_map(
        total_meas, x_gauge_coords + flag_coords + z_gauge_coords
    )
    rec_lookup_z_tail = {c: rec_lookup_tail[c] for c in z_gauge_coords}
    rec_lookup_x_gauge_tail = {c: rec_lookup_tail[c] for c in x_gauge_coords}

    if memory_in_x_basis:
        for strip in range(d - 1):
            col = 2 * strip + 1
            coords_in_strip = [c for c in x_gauge_coords if c[0] == col]
            if not coords_in_strip:
                continue
            detectors: List[int] = []
            for data_coord in x_strip_data_coords(strip):
                detectors.append(-data_rec_map[data_coord])
            for measure in coords_in_strip:
                detectors.append(-(len(data_qubits) + rec_lookup_x_gauge_tail[measure]))
            tail.append_operation(
                "DETECTOR",
                [stim.target_rec(x) for x in detectors],
                [col, 2 * (d - 1), 1.0],
            )
    else:
        z_order = [(0, 1), (0, -1)]
        for i in range(d - 1):
            for j in range(d - 1):
                if (i + j) % 2 != 0:
                    z_stab = [(2 * i, 2 * j + 1), (2 * i + 2, 2 * j + 1)]
                    detectors = []
                    for measure in z_stab:
                        if measure in qubits_cord_seq:
                            for delta in z_order:
                                data = tuple(np.add(measure, delta))
                                detectors.append(-data_rec_map[data])
                    for measure in z_stab:
                        detectors.append(-(len(data_qubits) + rec_lookup_z_tail[measure]))
                    tail.append_operation(
                        "DETECTOR",
                        [stim.target_rec(x) for x in detectors],
                        [2 * i + 1, 2 * j + 1, 1.0],
                    )
        for m in range(int((d - 1) / 2)):
            b_left = (0, 4 * m + 1)
            b_right = (2 * (d - 1), 4 * m + 3)
            for edge in [b_left, b_right]:
                if edge in rec_lookup_z_tail:
                    detectors = []
                    for delta in z_order:
                        data = tuple(np.add(edge, delta))
                        detectors.append(-data_rec_map[data])
                    detectors.append(-(len(data_qubits) + rec_lookup_z_tail[edge]))
                    tail.append_operation(
                        "DETECTOR",
                        [stim.target_rec(x) for x in detectors],
                        [edge[0], edge[1], 1.0],
                    )

    # Logical observable: first column (Z-basis) or first row (X-basis)
    logical_coords = (
        [c for c in data_coords if c[0] == 0]
        if memory_in_x_basis
        else [c for c in data_coords if c[1] == 0]
    )
    obs_targets = [stim.target_rec(-data_rec_map[c]) for c in logical_coords]
    obs_targets.sort(key=lambda t: t.value, reverse=True)
    tail.append_operation("OBSERVABLE_INCLUDE", obs_targets, 0)

    return head + body + tail


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------

def generate_circuit(
    *,
    rounds: int,
    distance: int,
    memory_basis: str = "Z",
    **noise_kwargs,
) -> stim.Circuit:
    """Build a heavy-hex memory-experiment circuit.

    Parameters
    ----------
    rounds : int
        Number of syndrome extraction rounds.
    distance : int
        Code distance.
    memory_basis : {"X", "Z"}
        Logical memory basis (default "Z").
    **noise_kwargs
        Forwarded to :class:`Circuitgen`.  Supported keys:

        * ``after_clifford_depolarization``
        * ``before_round_data_depolarization``
        * ``before_measure_flip_probability``
        * ``after_reset_flip_probability``

    Returns
    -------
    stim.Circuit
    """
    params = Circuitgen(rounds=rounds, d=distance, **noise_kwargs)
    return generate_heavy_hex_circuit(params, memory_basis=memory_basis)
