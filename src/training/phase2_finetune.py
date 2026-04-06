import os

from src.helper import utils
from src.config import paths
from src.helper.prompts import EMOTION_RECOGNITION_FINAL_PROMPT, GEMINI_EMOTION_RECOGNITION_PROMPT
from src.training.base_trainer import BaseTrainer
from datasets import DatasetDict, Dataset


os.environ["WANDB_PROJECT"] = "stage2-emotion-recognition-gemini"

class Phase2Trainer(BaseTrainer):
    """
    Trainer for Phase 2: Emotion Recognition.
    Inherits all common logic from BaseTrainer.
    """

    def __init__(self, config_dict=None):
        # Pass stage_id=2 to the BaseTrainer
        super().__init__(config_dict, stage_id=2)

        # Phase 2 MUST have a stage1_adapter_path
        # if not self.config.stage1_adapter_path:
        #     raise ValueError("--stage1_adapter_path is required for Phase 2 training.")

    def _get_default_args(self):
        """
        Returns a dictionary of the default hyperparameters specific
        to Phase 2 training.
        """
        return {
            "dataset": "iemocap",
            # "use_qlora": False,
            "iemocap_data_path": str(paths.TRAINING_STAGE2_DIR / "IEMOCAP"),
            "meld_data_path": str(paths.TRAINING_STAGE2_DIR / "MELD"),
            "stage1_adapter_path": "artifacts/finetuning/STAGE1-DEFAULT-r16/COMBINED/QLORA/final_checkpoint",
            "learning_rate": 5e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "early_stopping_patience": 2
        }


    # This function is used to create training set, if no set was provided via iemocap_data_path/meld_data_path
    def create_processed_data(self, dataset_name):
        from src.helper import build_prompting_dataset
        vectordb_path = utils.get_vectordb_path_from_attributes("hybrid", max_m=7)
        TRAINING_SET_CONFIGS = {"dataset": "iemocap", "max_k": 20, "top_n": 2, "split": ["train", "dev"],
                                "vectordb_path": vectordb_path, "use_detailed_example": True,
                                "example_refinement_level": 1, "save_as": "no"}
        tmp_configs = {**TRAINING_SET_CONFIGS, "dataset": dataset_name}
        processed_dataset = build_prompting_dataset.main(tmp_configs)  # {"train": [...], "dev": [...]}
        return processed_dataset


    def get_raw_datasets(self):
        """
        Implements the data loading logic specific to Phase 2.
        """
        # which datasets to use
        datasets_to_use = (
            ["meld", "iemocap"] if self.config.dataset == "both"
            else [self.config.dataset]
        )

        splits = ["train", "dev"]
        training_set = {split: [] for split in splits}

        for ds_name in datasets_to_use:
            # build the per-dataset candidate emotions text
            candidate_emotions = utils.get_mapped_emotion_set(ds_name)
            candidate_emotions_text = ", ".join(candidate_emotions)

            # if the training dataset was not created, create it
            if self.config.iemocap_data_path is None and self.config.meld_data_path is None:
                processed_dataset = self.create_processed_data(ds_name)
            else: # use prepared dataset
                config_dict = vars(self.config)
                train_proc = utils.load_json_multiline(os.path.join(config_dict[f"{ds_name}_data_path"], "train.jsonl"))
                dev_proc = utils.load_json_multiline(os.path.join(config_dict[f"{ds_name}_data_path"], "dev.jsonl"))
                processed_dataset = {"train": train_proc, "dev": dev_proc}

            # transform + merge
            for split in training_set.keys():
                transformed = []
                for ex in processed_dataset[split]:
                    inp = ex["input"].copy()
                    inp["candidate_emotions"] = candidate_emotions_text
                    transformed.append(
                        {
                            "inputs": inp,  # renamed from "input"
                            "output": ex["target"],
                            # "idx": ex["idx"],
                        }
                    )
                training_set[split].extend(transformed)

        # build HF DatasetDict
        raw_datasets = DatasetDict(
            {
                split: Dataset.from_list(training_set[split])
                for split in training_set.keys()
            }
        )
        return raw_datasets

    def load_and_prepare_data(self):
        """
        Implements the data loading and tokenization logic specific to Phase 2.

        NOTE: This assumes your RAG examples (retrieved_example) and
        candidate_emotions are already pre-processed and included in the
        .jsonl files under the 'inputs' key.
        """

        pass

    def get_prompt_template(self):
        assert self.config.prompt_type.lower() in ["gemini", "default", None]
        if self.config.prompt_type.lower() == "gemini":
            return GEMINI_EMOTION_RECOGNITION_PROMPT
        return EMOTION_RECOGNITION_FINAL_PROMPT


# --- Script Entry Point ---
if __name__ == "__main__":
    # When run directly, config_dict is None, so it will parse command-line args
    # The parser will use the defaults from _get_default_args
    print("Phase 2 training is started!")
    trainer = Phase2Trainer(config_dict=None)
    trainer.run()
