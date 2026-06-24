"""
Generate heavy-hexagon syndrome datasets and save them as .npy files.

Usage
-----
python scripts/generate_data.py \\
    --distance 5 \\
    --rounds_list 3 5 7 9 11 13 15 \\
    --num_shots 10000000 \\
    --noise_max 1e-3 \\
    --basis Z \\
    --outdir /path/to/output

Output files (per rounds value r)
----------------------------------
detection_{basis_tag}_d{distance}_r{r}.npy   -- bool array (shots, num_detectors)
observable_{basis_tag}_d{distance}_r{r}.npy  -- bool array (shots, 1)

where basis_tag is "zmem" or "xmem".
"""

import argparse
import os

import numpy as np

from hhd.circuit.hh_circ import generate_circuit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heavy-hex syndrome datasets")
    parser.add_argument("--distance",    type=int,   required=True,
                        help="Code distance (e.g. 5)")
    parser.add_argument("--rounds_list", type=int,   nargs="+",
                        default=[3, 5, 7, 9, 11, 13, 15],
                        help="List of syndrome-round values to generate")
    parser.add_argument("--num_shots",   type=int,   default=10_000_000,
                        help="Number of Stim shots per (distance, rounds) pair")
    parser.add_argument("--noise_max",   type=float, default=1e-3,
                        help="Physical error probability p (all four noise params)")
    parser.add_argument("--basis",       type=str,   default="Z",
                        choices=["X", "Z"],
                        help="Memory experiment basis")
    parser.add_argument("--outdir",      type=str,
                        default=os.getenv("SCRATCH", "."),
                        help="Output directory for .npy files")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    p = args.noise_max
    basis = args.basis.upper()
    basis_tag = "xmem" if basis == "X" else "zmem"

    for rounds in args.rounds_list:
        print(f"\n[d={args.distance}, r={rounds}, basis={basis}] generating {args.num_shots:,} shots ...")

        circ = generate_circuit(
            rounds=rounds,
            distance=args.distance,
            memory_basis=basis,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )

        sampler = circ.compile_detector_sampler()
        dets, obs = sampler.sample(shots=args.num_shots, separate_observables=True)

        tag = f"{basis_tag}_d{args.distance}_r{rounds}"
        det_path = os.path.join(args.outdir, f"detection_{tag}.npy")
        obs_path = os.path.join(args.outdir, f"observable_{tag}.npy")

        np.save(det_path, dets)
        np.save(obs_path, obs)

        print(f"  detection shape : {dets.shape}")
        print(f"  observable shape: {obs.shape}")
        print(f"  saved -> {det_path}")
        print(f"  saved -> {obs_path}")

    print("\nAll data saved.")


if __name__ == "__main__":
    main()
