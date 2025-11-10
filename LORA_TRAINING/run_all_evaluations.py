import os
import glob
import argparse
import subprocess


def find_adapter_paths(finetuning_root: str):
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
        os.path.join(finetuning_root, "**", "final_checkpoint*"),
        os.path.join(finetuning_root, "**", "final_model*"),
    ]

    adapter_paths = set()

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isdir(path):
                adapter_paths.add(os.path.normpath(path))

    return sorted(adapter_paths)


def run_single_eval(
    eval_script: str,
    base_model_id: str,
    adapter_path: str | None,
    split: str,
    use_qlora: bool,
    limit: int | None,
):
    """
    Call the single-model evaluation script with the given parameters.
    If adapter_path is None, evaluates the base model only.
    """
    cmd = [
        "python",
        eval_script,
        "--base_model_id",
        base_model_id,
        "--split",
        split,
        "--use_qlora",
        "true" if use_qlora else "false",
    ]

    if adapter_path is not None:
        cmd.extend(["--adapter_path", adapter_path])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

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
        # from evaluate_single_model.py (which exits with non-zero status).
        print(f"❌ Evaluation failed for adapter={adapter_path}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate_single_model.py for base model and all adapters."
    )

    parser.add_argument(
        "--eval_script",
        type=str,
        default="evaluate_single_model.py",
        help="Path to the single-model evaluation script.",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model ID to pass to the evaluation script.",
    )
    parser.add_argument(
        "--finetuning_root",
        type=str,
        default="FINETUNING",
        help="Root directory where LoRA checkpoints/final models are stored.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
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
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of examples to evaluate.",
    )
    parser.add_argument(
        "--skip_base",
        type=lambda x: (str(x).lower() in ["true", "1", "t", "yes", "y"]),
        default=False,
        help="If True, do NOT run the base model evaluation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Run base model (if not skipped)
    if not args.skip_base:
        run_single_eval(
            eval_script=args.eval_script,
            base_model_id=args.base_model_id,
            adapter_path=None,
            split=args.split,
            use_qlora=args.use_qlora,
            limit=args.limit,
        )
    else:
        print("Skipping base model evaluation (skip_base=True).")

    # 2) Find all adapter checkpoints/final models
    print(f"\nSearching for adapters under: {args.finetuning_root}")
    adapter_paths = find_adapter_paths(args.finetuning_root)
    print(f"Found {len(adapter_paths)} adapter directories.")

    # 3) Evaluate each adapter
    for adapter_path in adapter_paths:
        run_single_eval(
            eval_script=args.eval_script,
            base_model_id=args.base_model_id,
            adapter_path=adapter_path,
            split=args.split,
            use_qlora=args.use_qlora,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
