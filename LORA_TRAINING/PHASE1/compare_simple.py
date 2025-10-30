"""
Simple Model Comparison - No Extra Dependencies Required
Compares Base vs Fine-tuned Model on Dev Dataset
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import json
import os
from typing import List, Dict
import argparse
from tqdm import tqdm
from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE
import utils
from transformers import BitsAndBytesConfig


class SimpleModelComparator:
    """Simple comparator using only basic metrics"""

    def __init__(self,
                 base_model_name: str,
                 finetuned_model_path: str,
                 device: str = "auto",
                 use_qlora: bool = True):
        """
        Initialize comparator

        Args:
            base_model_name: Name of base model
            finetuned_model_path: Path to fine-tuned LoRA adapter
            device: Device to use
        """
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        self.use_qlora = use_qlora

        print("=" * 80)
        print("MODEL COMPARISON SETUP")
        print("=" * 80)
        print(f"Base Model: {base_model_name}")
        print(f"Fine-tuned Model: {finetuned_model_path}")
        print(f"Fine-tuned Model Quantization: {self.use_qlora}")  # Print setting
        print("=" * 80)

        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print("\nLoading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        self.base_model.eval()

        # Load fine-tuned model
        print(f"\nLoading fine-tuned model (Quantized: {self.use_qlora})...")
        load_kwargs = {}

        if self.use_qlora:
            # 4-bit configuration for QLoRA-trained models
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,  # Use bfloat16 if your GPU supports it
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs['quantization_config'] = quantization_config

            # For QLoRA models, device_map='auto' is often required/better
            if device == "auto":
                load_kwargs['device_map'] = "auto"
            print("  -> Using 4-bit QLoRA loading configuration.")
        else:
            # Standard FP16/BF16 loading (non-quantized)
            load_kwargs['torch_dtype'] = torch.float16
            load_kwargs['device_map'] = device

        base_for_ft = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **load_kwargs
        )

        self.finetuned_model = PeftModel.from_pretrained(
            base_for_ft,
            finetuned_model_path
        )
        self.finetuned_model.eval()

        print("\nModels loaded successfully!")

    def create_prompt(self, history: str, speaker_id: str,
                      utterance: str, audio_features: str) -> str:
        """Create prompt for model"""

        return SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE.format(
            history=history,
            speaker_id=speaker_id,
            utterance=utterance,
            audio_features=audio_features
        )

    def generate_response(self, model, prompt: str, max_new_tokens: int = 15) -> str:
        """Generate response from model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def word_overlap_score(self, prediction: str, reference: str) -> float:
        """Calculate simple word overlap score"""
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())

        if len(ref_words) == 0:
            return 0.0

        overlap = len(pred_words & ref_words)
        return overlap / len(ref_words)

    def exact_match_score(self, prediction: str, reference: str) -> float:
        """Check if prediction exactly matches reference"""
        return 1.0 if prediction.lower().strip() == reference.lower().strip() else 0.0

    def length_penalty(self, prediction: str, max_words: int = 15) -> float:
        """Penalize outputs that are too long or too short"""
        num_words = len(prediction.split())
        if num_words == 0:
            return 0.0
        elif num_words <= max_words:
            return 1.0
        else:
            return max(0.0, 1.0 - (num_words - max_words) / max_words)

    def compare_on_example(self, example: Dict) -> Dict:
        """Compare both models on a single example"""
        # Extract fields
        inputs = example['inputs']
        ground_truth = example['output']

        # Create prompt
        prompt = self.create_prompt(
            history=inputs['history'],
            speaker_id=inputs['speaker_id'],
            utterance=inputs['utterance'],
            audio_features=inputs['audio_features']
        )

        # Generate from base model
        base_response = self.generate_response(self.base_model, prompt)

        # Generate from fine-tuned model
        ft_response = self.generate_response(self.finetuned_model, prompt)

        # Calculate metrics
        base_overlap = self.word_overlap_score(base_response, ground_truth)
        ft_overlap = self.word_overlap_score(ft_response, ground_truth)

        base_exact = self.exact_match_score(base_response, ground_truth)
        ft_exact = self.exact_match_score(ft_response, ground_truth)

        base_length = self.length_penalty(base_response)
        ft_length = self.length_penalty(ft_response)

        return {
            'example_id': example.get('iden', 'unknown'),
            'ground_truth': ground_truth,
            'base_response': base_response,
            'finetuned_response': ft_response,
            'base_scores': {
                'word_overlap': base_overlap,
                'exact_match': base_exact,
                'length_penalty': base_length
            },
            'ft_scores': {
                'word_overlap': ft_overlap,
                'exact_match': ft_exact,
                'length_penalty': ft_length
            },
            'winner': 'finetuned' if ft_overlap > base_overlap else 'base' if base_overlap > ft_overlap else 'tie'
        }

    def evaluate_on_dataset(self, dataset, num_samples: int = None, verbose=True) -> Dict:
        """
        Evaluate both models on dataset

        Args:
            dataset: HuggingFace dataset
            num_samples: Number of samples to evaluate (None = all)

        Returns:
            Dictionary with comparison results
        """
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        print(f"\nEvaluating on {len(dataset)} examples...")

        results = []
        base_metrics = {'word_overlap': [], 'exact_match': [], 'length_penalty': []}
        ft_metrics = {'word_overlap': [], 'exact_match': [], 'length_penalty': []}

        for example in tqdm(dataset, desc="Comparing models"):
            # Compare models on this example
            comparison = self.compare_on_example(example)
            results.append(comparison)

            if verbose:
                tqdm.write("\n" + "=" * 50)
                tqdm.write(f"Example ID: {comparison['example_id']}")
                tqdm.write(f"  Ground Truth:     {comparison['ground_truth']}")
                tqdm.write(f"  Base Response:    {comparison['base_response']}")
                tqdm.write(f"  Finetuned Response: {comparison['finetuned_response']}")
                tqdm.write("=" * 50)

            # Accumulate metrics
            for metric in base_metrics:
                base_metrics[metric].append(comparison['base_scores'][metric])
                ft_metrics[metric].append(comparison['ft_scores'][metric])

        # Calculate averages
        avg_base = {metric: sum(scores) / len(scores) for metric, scores in base_metrics.items()}
        avg_ft = {metric: sum(scores) / len(scores) for metric, scores in ft_metrics.items()}

        # Count wins
        wins = {'base': 0, 'finetuned': 0, 'tie': 0}
        for result in results:
            wins[result['winner']] += 1

        return {
            'individual_results': results,
            'base_model_avg': avg_base,
            'finetuned_model_avg': avg_ft,
            'wins': wins,
            'num_samples': len(results)
        }

    def print_comparison_report(self, eval_results: Dict):
        """Print detailed comparison report"""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)

        print(f"\nNumber of samples evaluated: {eval_results['num_samples']}")

        print("\n" + "-" * 80)
        print("AVERAGE SCORES")
        print("-" * 80)

        print("\nBase Model:")
        for metric, score in eval_results['base_model_avg'].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.4f}")

        print("\nFine-tuned Model:")
        for metric, score in eval_results['finetuned_model_avg'].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.4f}")

        print("\n" + "-" * 80)
        print("IMPROVEMENT")
        print("-" * 80)
        for metric in eval_results['base_model_avg']:
            base_score = eval_results['base_model_avg'][metric]
            ft_score = eval_results['finetuned_model_avg'][metric]
            if base_score > 0:
                improvement = ((ft_score - base_score) / base_score) * 100
                print(f"  {metric.replace('_', ' ').title()}: {improvement:+.2f}%")

        print("\n" + "-" * 80)
        print("WIN/LOSS/TIE")
        print("-" * 80)
        total = eval_results['num_samples']
        wins = eval_results['wins']
        print(f"  Fine-tuned Wins: {wins['finetuned']} ({wins['finetuned'] / total * 100:.1f}%)")
        print(f"  Base Wins: {wins['base']} ({wins['base'] / total * 100:.1f}%)")
        print(f"  Ties: {wins['tie']} ({wins['tie'] / total * 100:.1f}%)")

        print("\n" + "=" * 80)

    def save_detailed_results(self, eval_results: Dict,
                              output_path: str,
                              num_examples: int = 20):
        """Save detailed results to file and print samples"""
        # Save full results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        print(f"\nFull results saved to: {output_path}")

        # Print sample comparisons
        print("\n" + "=" * 80)
        print(f"SAMPLE COMPARISONS (First {num_examples} examples)")
        print("=" * 80)

        for i, result in enumerate(eval_results['individual_results'][:num_examples]):
            print(f"\n{'-' * 80}")
            print(f"Example {i + 1} (ID: {result['example_id']})")
            print(f"{'-' * 80}")

            print(f"\nGround Truth:")
            print(f"  {result['ground_truth']}")

            print(f"\nBase Model:")
            print(f"  Output: {result['base_response']}")
            print(f"  Word Overlap: {result['base_scores']['word_overlap']:.3f}")

            print(f"\nFine-tuned Model:")
            print(f"  Output: {result['finetuned_response']}")
            print(f"  Word Overlap: {result['ft_scores']['word_overlap']:.3f}")

            print(f"\n  Winner: {result['winner'].upper()}")


def load_dev_dataset(data_path: str):
    """Load development dataset"""
    # data_files = {"dev": os.path.join(data_path, "dev.jsonl")}
    # dataset = load_dataset("json", data_files=data_files)
    # return dataset["dev"]


    data_path = os.path.join(utils.PROJECT_PATH, "TRAINING_DATA/PHASE1/IEMOCAP/")

    data_files = {
        # "train": os.path.join(data_path, "train.jsonl"),
        "dev": os.path.join(data_path, "dev.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)
    # return raw_datasets["train"], raw_datasets["dev"]
    return raw_datasets["dev"]


def main():
    parser = argparse.ArgumentParser(
        description='Compare Base and Fine-tuned Models (Simple Version)'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Base model name'
    )
    parser.add_argument(
        '--finetuned_model',
        type=str,
        default='FINETUNING/PHASE1_A/checkpoint-200/',
        help='Path to fine-tuned model'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='TRAINING_DATA/PHASE1/IEMOCAP/',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='comparison_results2.json',
        help='Output file for results'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=300,
        help='Number of samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--num_display',
        type=int,
        default=20,
        help='Number of examples to display'
    )

    parser.add_argument(  # <-- NEW ARGUMENT
        '--use_qlora',
        type=lambda x: (str(x).lower() in ['true', '1', 't']),  # Allows 'True', 'true', '1', etc.
        default=True,
        help='Whether to load the fine-tuned model base using 4-bit quantization (required for QLoRA).'
    )

    args = parser.parse_args()

    # Load dev dataset
    print(f"Loading dev dataset from: {args.data_dir}")
    dev_dataset = load_dev_dataset(args.data_dir)
    print(f"Loaded {len(dev_dataset)} examples")

    # Initialize comparator
    comparator = SimpleModelComparator(
        base_model_name=args.base_model,
        finetuned_model_path=args.finetuned_model,
        use_qlora=args.use_qlora
    )

    # Evaluate
    eval_results = comparator.evaluate_on_dataset(
        dev_dataset,
        num_samples=args.num_samples
    )

    # Print report
    comparator.print_comparison_report(eval_results)

    # Save results
    comparator.save_detailed_results(
        eval_results,
        args.output_file,
        num_examples=args.num_display
    )

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()