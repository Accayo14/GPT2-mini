from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-2 small toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="prepare tokenized dataset files")
    subparsers.add_parser("train", help="train model")
    subparsers.add_parser("sample", help="generate text")
    subparsers.add_parser("decompose", help="decompose logits into residual components")
    args, rest = parser.parse_known_args()

    if args.command == "prepare":
        try:
            from .prepare_dataset import main as prepare_main
        except ImportError:
            from prepare_dataset import main as prepare_main
        prepare_main(rest)
    elif args.command == "train":
        try:
            from .train import main as train_main
        except ImportError:
            from train import main as train_main
        train_main(rest)
    elif args.command == "sample":
        try:
            from .sample import main as sample_main
        except ImportError:
            from sample import main as sample_main
        sample_main(rest)
    elif args.command == "decompose":
        try:
            from .decompose import main as decompose_main
        except ImportError:
            from decompose import main as decompose_main
        decompose_main(rest)


if __name__ == "__main__":
    main()
