#!/usr/bin/env python3.11
""""""
import argparse
import pathlib as p
import attractors
from main import AttractorPlotter, read_gpl


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an attractor.")
    parser.add_argument(
        "attractor",
        metavar="A",
        type=str,
        help="the attractor to render",
        choices=["clifford_attractor", "modified_clifford_attractor", "jason_rampe_3_attractor"],
    )
    parser.add_argument(
        "--palette", metavar="P", type=p.Path, help="the palette to use", default=p.Path("palettes/ds8.gpl")
    )
    parser.add_argument("--output", metavar="O", type=p.Path, help="the output file to write to")
    parser.add_argument("--params", type=float, help="parameteres for the attractor", nargs="*")
    parser.add_argument("--seed", type=int, help="the random seed", default=None)
    parser.add_argument("--size", type=int, help="the size of the rendered image", default=2000)
    parser.add_argument("--samples", type=int, help="the number of samples to iterate over", default=20_000)
    parser.add_argument("--iterations", type=int, help="the number of iterations to do per sample", default=12)

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    assert args.palette.suffix == ".gpl"
    assert args.palette.exists()

    with args.palette.open("r", encoding="utf-8") as file:
        palette = read_gpl(file)

    attractor = getattr(attractors, args.attractor)(*args.params)
    AttractorPlotter(
        attractor, scale=0.6, palette=palette, size=args.size, n_samples=args.samples, n_iterations=args.iterations
    ).render_to(args.output)


if __name__ == "__main__":
    main()
