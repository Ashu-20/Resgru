# Heavy-Hexagon Circuit Layout

## Qubit Grid

The heavy-hexagon code is defined on a 2D grid with the following coordinate
conventions used in this codebase:

- **Data qubits** sit at even coordinates `(2i, 2j)` for `0 ≤ i, j < d`.
- **Z-gauge ancillas** sit at `(2i, 2j+1)` — horizontally between adjacent data qubits on the same row.
- **X-gauge ancillas** (bulk, weight-4) sit at `(2i+1, 2j+1)` for `(i+j) % 2 == 0`.
- **X-gauge ancillas** (boundary, weight-2) sit along the top and bottom edges.
- **Flag qubits** sit at `(2i, 2j+1)` and `(2i+2, 2j+1)` adjacent to each bulk X-gauge.

All qubit indices are computed row-major: `index = (2d-1)*row + col`.

## Syndrome Extraction Order

Each round extracts X-type and Z-type gauge measurements:

1. X-gauge extraction uses a 5-step CNOT schedule through flag qubits, followed by `MRX` on X and flag ancillas.
2. Z-gauge extraction uses a 3-step CNOT schedule, followed by `MRZ` on Z ancillas.

## Detector Layout

**Z-type detectors** are formed from pairs of neighbouring Z-gauges on
adjacent rows: `(2i, 2j+1)` and `(2i+2, 2j+1)` where `(i+j) % 2 != 0`.
These correspond to the weight-4 Z stabilizers of the underlying surface code.

**X-type detectors** (Bacon-Shor strips) are formed from all X-gauges in a
given column (vertical strip), XORed with the same strip in the previous round.

## Logical Observable

- **Z-basis memory**: parity of the leftmost column of data qubits (`col == 0`).
- **X-basis memory**: parity of the top row of data qubits (`row == 0`).

## Syndrome Tensor Layout

For a circuit with `r` rounds at distance `d`:

```
num_z   = (d-1) + (d-1)^2 / 2         # Z-type gauges per round
dpr     = (d-1)(d+3) / 2              # detectors per round (X + Z)

syndrome row shape: (num_z + (r-1)*dpr + num_z,)
                     ^init    ^core rounds   ^final
```

The training scripts slice this as:
- `init  = row[:num_z]`
- `core  = row[num_z:-num_z].reshape(r-1, dpr)`
- `final = row[-num_z:]`

`core` is the variable-length input to the GRU; `init` and `final` are
broadcast to every time step as side information.
