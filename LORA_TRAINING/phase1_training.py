import os
import utils
from prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE
from LORA_TRAINING.training_helper import BaseTrainer
from datasets import DatasetDict, Dataset


os.environ["WANDB_PROJECT"] = "stage1-speaker-extraction"

class Phase1Trainer(BaseTrainer):
    """
    Trainer for Phase 1: Speaker Characteristic Injection Fine-Tuning.
    Inherits all common logic from BaseTrainer.
    """

    def __init__(self, config_dict=None):
        # Pass stage_id=2 to the BaseTrainer
        super().__init__(config_dict, stage_id=1)


    def _get_default_args(self):
        """
        Returns a dictionary of the default hyperparameters specific
        to Phase 1 training.
        """
        return {
            "dataset": "combined",
            "iemocap_data_path": f"TRAINING_DATA/PHASE{self.stage_id}/IEMOCAP/",
            "meld_data_path": f"TRAINING_DATA/PHASE{self.stage_id}/MELD/",
            "learning_rate": 2e-4
        }




    def get_raw_datasets(self):
        """
        Implements the data loading logic specific to Phase 1.
        """

        data_dirs = self.get_data_dirs()
        raw_datasets = self._load_data_from_dirs(data_dirs)
        return raw_datasets

    def get_prompt_template(self):
        return SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE



# --- Script Entry Point ---
if __name__ == "__main__":
    # When run directly, config_dict is None, so it will parse command-line args
    # The parser will use the defaults from _get_default_args
    trainer = Phase1Trainer(config_dict=None)
    trainer.run()

