import os

from src.helper import utils
from src.config import paths
from src.helper.prompts import SPEAKER_CHARACTERISTICS_EXTRACTION_TEMPLATE
from src.training.base_trainer import BaseTrainer


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
            "dataset": "both",
            "iemocap_data_path": str(paths.TRAINING_STAGE1_DIR / "IEMOCAP"),
            "meld_data_path": str(paths.TRAINING_STAGE1_DIR / "MELD"),
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
