"""Runner script that evaluates all Phase 2 adapter checkpoints in sequence.

Discovers every checkpoint directory under a given ``--finetuning_root``
and calls :mod:`stage2_eval_parallel` as a subprocess for each one.
Optionally also evaluates the base model without any adapter.

Supports evaluating on MELD, IEMOCAP, or both datasets in a single run.

Usage::

    python -m src.training.run_full_stage2_evaluations \\
        --finetuning_root artifacts/finetuning/STAGE1_2-DEFAULT-r16 \\
        --dataset both --split test [--skip_base true]
"""

import os
import glob
import argparse
import subprocess
from typing import Optional

from src.config import constants


def find_adapter_paths(finetuning_root: str):
    """Recursively find all checkpoint directories under a fine-tuning root.

    Matches directories named ``checkpoint-*`` (intermediate checkpoints
    saved during training).

    Args:
        finetuning_root (str): Root directory to search.

    Returns:
        list[str]: Sorted list of absolute checkpoint directory paths.
    """
    """
    Find all adapter directories under `finetuning_root`.

    Matches:
      - **/checkpoint-*
      - **/final_checkpoint*
      - **/final_model*

    Returns a sorted list of absolute paths.
    """
    patterns = [
        os.path.join(finetuning_root, "**", "checkpoint-*"),
        # os.path.join(finetuning_root, "**", "final_checkpoint*"),
    ]

    adapter_paths = set()

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                adapter_paths.add(os.path.normpath(path))

    return sorted(adapter_paths)


def run_single_eval(
    eval_module: str,
    base_model_id: str,
    adapter_path: Optional[str],
    dataset: str,
    split: str,
    use_qlora: bool,
    limit: Optional[int],
    verbose: Optional[int]
):
    """
    Call the single-model evaluation script (as a module) with the given parameters.

    If adapter_path is None, evaluates the base model only.

    This runs:
        python -m <eval_module> ...
    e.g.:
        python -m src.training.stage2_eval_parallel --base_model_id ...
    """
    cmd = [
        "python",
        "-m",
        eval_module,
        "--base_model_id",
        base_model_id,
        "--dataset",
        dataset,
        "--split",
        split,
        "--use_qlora",
        "true" if use_qlora else "false",

    ]

    if adapter_path is not None:
        cmd.extend(["--adapter_path", adapter_path])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if verbose is not None:
        cmd.extend(["--verbose", str(verbose)])

    print("\n============================================================")
    if adapter_path:
        print(f"Running evaluation for adapter:\n  {adapter_path}")
    else:
        print("Running evaluation for BASE model")
    print("Command:", " ".join(cmd))
    print("============================================================\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # This will also catch the “results already exist” RuntimeError
        print(f"❌ Evaluation failed for adapter={adapter_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate_single_phase2_model_parallel.py for base model and all adapters."
    )

    parser.add_argument(
        "--eval_module",
        type=str,
        default="llm_for_erc.training.stage2_eval_parallel",
        help="Module path of the single-model evaluation script (used with `python -m`).",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default=constants.BASE_LLM_MODEL,
        help="Base model ID to pass to the evaluation script.",
    )
    parser.add_argument(
        "--finetuning_root",
        type=str,
        # default="artifacts/finetuning/STAGE1_2-DEFAULT-r16/",
        help="Root directory where LoRA checkpoints/final models are stored.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["meld", "iemocap", "both"],
        help="which dataset to use for evaluation (iemocap, meld, both)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--use_qlora",
        type=lambda x: (str(x).lower() in ["true", "1", "t", "yes", "y"]),
        default=True,
        help="Whether to use QLoRA (4-bit) in the evaluation script.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0 = silent, 1 = show GT + extracted prediction + score, 2 = show full raw info."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of examples to evaluate.",
    )
    parser.add_argument(
        "--skip_base",
        type=lambda x: (str(x).lower() in ["true", "1", "t", "yes", "y"]),
        default=True,
        help="If True, do NOT run the base model evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.finetuning_root is None:
        raise Exception("--finetuning_root argument must be given which determines allows "
                        "the script to find the finetuned adapter weights")

    # 1) Run base model (if not skipped)
    if not args.skip_base:
        run_single_eval(
            eval_module=args.eval_module,
            base_model_id=args.base_model_id,
            adapter_path=None,
            dataset=args.dataset,
            split=args.split,
            use_qlora=args.use_qlora,
            limit=args.limit,
            verbose=args.verbose
        )
    else:
        print("Skipping base model evaluation (skip_base=True).")


    # 2) Find all adapter checkpoints/final models
    print(f"\nSearching for adapters under: {args.finetuning_root}")
    adapter_paths = find_adapter_paths(args.finetuning_root)
    print(f"Found {len(adapter_paths)} adapter directories.")

    # 3) Evaluate each adapter
    if args.dataset.lower() in ["meld", "combined", "both"]:
        for adapter_path in adapter_paths:
            run_single_eval(
                eval_module=args.eval_module,
                base_model_id=args.base_model_id,
                adapter_path=adapter_path,
                dataset="meld",
                split=args.split,
                use_qlora=args.use_qlora,
                limit=args.limit,
                verbose=args.verbose
            )
    if args.dataset.lower() in ["iemocap", "combined", "both"]:
        for adapter_path in adapter_paths:
            run_single_eval(
                eval_module=args.eval_module,
                base_model_id=args.base_model_id,
                adapter_path=adapter_path,
                dataset="iemocap",
                split=args.split,
                use_qlora=args.use_qlora,
                limit=args.limit,
                verbose=args.verbose
            )


if __name__ == "__main__":
    main()
