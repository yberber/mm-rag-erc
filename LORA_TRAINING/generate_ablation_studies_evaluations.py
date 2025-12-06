# This script is used to generate evaluation results with best
# IEMOCAP and MELD model in 3 different setting:
# 1) without RAG
# 2) without Audio
# 3) without RAG and Audio
# Later this results including base model results will be compared with our
# best model to see the improvement via different components

import argparse
import subprocess
from typing import Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluate_single_erc_model_parallel.py for base model and all adapters."
    )

    parser.add_argument(
        "--eval_module",
        type=str,
        default="LORA_TRAINING.evaluate_single_erc_model_parallel",
        help=(
            "Module path of the single-model evaluation script "
            "(used with `python -m`). "
            "Default: LORA_TRAINING.evaluate_single_model"
        ),
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
        default="FINETUNING/STAGE1_2-DEFAULT-r16/",
        help="Root directory where LoRA checkpoints/final models are stored.",
    )
    parser.add_argument(
        "--meld_adapter",
        type=str,
        default="FINETUNING/STAGE1_2-DEFAULT-r16/MELD/QLORA/checkpoint-750",
        help="Adapter to load for MELD evaluation"
    )
    parser.add_argument(
        "--iemocap_adapter",
        type=str,
        default="FINETUNING/STAGE1_2-DEFAULT-r16/IEMOCAP/QLORA/checkpoint-750",
        help="Adapter to load for MELD evaluation"
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

    return parser.parse_args()



def run_single_eval(
    eval_module: str,
    base_model_id: str,
    adapter_path: Optional[str],
    dataset: str,
    split: str,
    use_qlora: bool,
    limit: Optional[int],
    verbose: Optional[int],
    use_audio: bool,
    use_rag_in_context: bool
):
    """
    Call the single-model evaluation script (as a module) with the given parameters.

    If adapter_path is None, evaluates the base model only.

    This runs:
        python -m <eval_module> ...
    e.g.:
        python -m LORA_TRAINING.evaluate_single_erc_model_parallel --base_model_id ...
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
        "true" if use_audio else "false",
        "true" if use_rag_in_context else 'false'

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



def main():
    args = parse_args()



    print(f"MELD ADAPTER: {args.meld_adapter}")
    print(f"IEMOCAP ADAPTER: {args.iemocap_adapter}")
    print(f"DATASET: {args.dataset}")


    use_audio_list = [False, True, False]
    use_rag_list = [True, False, False]

    for (use_audio, use_rag) in zip(use_audio_list, use_rag_list):

        if args.dataset.lower() in ["meld", "combined", "both"]:
            run_single_eval(
                eval_module=args.eval_module,
                base_model_id=args.base_model_id,
                adapter_path=args.meld_adapter,
                dataset="meld",
                split=args.split,
                use_qlora=args.use_qlora,
                limit=args.limit,
                verbose=args.verbose,
                use_audio=use_audio,
                use_rag_in_context=use_rag
            )
        if args.dataset.lower() in ["iemocap", "combined", "both"]:
            run_single_eval(
                eval_module=args.eval_module,
                base_model_id=args.base_model_id,
                adapter_path=args.iemocap_adapter,
                dataset="iemocap",
                split=args.split,
                use_qlora=args.use_qlora,
                limit=args.limit,
                verbose=args.verbose,
                use_audio=use_audio,
                use_rag_in_context=use_rag
            )



if __name__ == "__main__":
    main()
