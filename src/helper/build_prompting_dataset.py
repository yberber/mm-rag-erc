import argparse
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from src.helper import utils
from src.config import paths

DEFAULT_VECTORSTORE_PATH = str(paths.VECTORSTORE_DB_DIR / "meld_iemocap_single")

class DemonstrationCreatorViaCache:
    def __init__(self, vectordb_path, top_n=1, use_detailed_example=False, example_refinement_level=0, valid_emotion_set=None):

        self.vectordb_path = Path(vectordb_path)
        cache_file = paths.VECTORSTORE_CACHE_DIR / f"{self.vectordb_path.name}.json"
        self.cache_data = utils.load_json(path=cache_file)
        self.top_n = top_n

        self.check_from_same_dataset = (example_refinement_level > 0)
        self.valid_emotion_set = valid_emotion_set

        vectordb_name = self.vectordb_path.name
        self.db_type = vectordb_name.split("_")[2]

        if self.db_type in ["flow", "hybrid"]:
            self.max_m = int(vectordb_name.split("_")[-1])
        else:
            self.max_m = 1

        self.use_detailed_example = use_detailed_example
        self.idx_to_speaker_utterance_emotion = self.get_idx_to_speaker_utterance_emotion_df()

    def get_id(self):
        return f"{self.db_type}{'V2' if self.use_detailed_example and self.db_type in ['flow', 'hybrid'] else ''}_n{self.top_n}_m{self.max_m}"

    def get_demonstration_text_via_idx(self, idx):
        top_n = self.retrieve_example_idx(idx)
        final_demonstration_text = []
        for ex_idx in top_n:
            idx_prefix = ex_idx[:ex_idx.rfind("_")]
            idx_turn_id = int(ex_idx.split("_")[-1])
            start_idx = idx_prefix + "_" + str(max(0, idx_turn_id + 1 - self.max_m))
            demonstration_text = self.get_demonstration_text_for_n1(self.idx_to_speaker_utterance_emotion,
                                                                   ex_idx, start_idx, type=self.db_type,
                                                                    use_detailed_example=self.use_detailed_example,
                                                                    valid_emotion_set=self.valid_emotion_set)
            final_demonstration_text.append(demonstration_text)

        if self.db_type == "single" or self.top_n <= 1:
            return "\n".join(final_demonstration_text)
        else:
            final_text = ""
            for i, demonstration in enumerate(final_demonstration_text, start=1):
                final_text += f"Demonstration {i}:\n{demonstration}\n\n"
            return final_text.strip()


    def retrieve_example_idx(self, idx = None):
        examples_idx = self.cache_data[idx]

        top_n = []
        excluded_prefixes = set()

        required_prefix = None
        if self.check_from_same_dataset and idx is not None:
            excluded_prefixes.add(idx[:idx.rfind("_")])
            required_prefix = idx.split("_")[0]

        for ex_idx in examples_idx:
            idx_prefix = ex_idx[:ex_idx.rfind("_")]
            if (idx_prefix not in excluded_prefixes) and (required_prefix is None or ex_idx.startswith(required_prefix)):
                top_n.append(ex_idx)
                if len(top_n) >= self.top_n:
                    break
                excluded_prefixes.add(idx_prefix)

        return top_n

    def get_idx_to_speaker_utterance_emotion_df(self):
        return utils.load_dataframe_from_json(path=paths.VECTORSTORE_INDEX_PATH)

    @staticmethod
    def get_demonstration_text_for_n1(df, turn_idx_target, turn_idx_start=None, type="single", use_detailed_example=False, valid_emotion_set=None):
        if type == "single":
            row = df.loc[turn_idx_target]
            return f'< {row["utterance"]} : {row["mapped_emotion"]} >'
        elif type in ["flow", "hybrid"]:
            df = df.loc[turn_idx_start:turn_idx_target]
            page_content_lines = []
            for _, row in df.iterrows():
                line = f"{row['speaker']} : {row['utterance']}"
                if use_detailed_example:
                    if valid_emotion_set and row['mapped_emotion'] not in valid_emotion_set:
                        continue
                    line += f" :-> {row['mapped_emotion']}"
                page_content_lines.append(line)
            demonstration = "\n".join(page_content_lines)
            if use_detailed_example:
                return f"<\n{demonstration}\n>"
            else:
                return f"< {demonstration}\nLast utterance label is: {df.iloc[-1]['mapped_emotion']} >"
        else:
            raise Exception(f"Unknown type: {type}")




def create_history_context(conversation, current_utterance_idx, max_k):
    context = ""
    conversation_history = conversation[max(0, current_utterance_idx - max_k):current_utterance_idx+1]
    for unit in conversation_history.to_dict(orient="records"):
        context += f"{unit['speaker']}: {unit['utterance']}\n"
    return context.strip("\n")



def get_vectordb_instance(path_to_db):
    from langchain_chroma import Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name=str(path_to_db).split("/")[-1],
        embedding_function=embeddings,
        persist_directory=path_to_db,
    )
    return vector_db



def process_split(split):
    if split is None:
        splits = ["train", "dev", "test"]
    elif isinstance(split, list):
        splits = split
    elif isinstance(split, str):
        splits = [split]
    else:
        raise ValueError(f"split must be either None or a list of strings, but got {split}")
    return splits

def get_valid_emotion_set_for_examples(dataset, example_refinement_level):
    valid_emotion_set_for_examples = None
    if example_refinement_level == 2:
        if dataset.lower() == "iemocap":
            valid_emotion_set_for_examples = utils.iemocap_mapped_valid_emotion_set
        elif dataset.lower() == "meld":
            valid_emotion_set_for_examples = utils.meld_mapped_valid_emotion_set
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    return valid_emotion_set_for_examples

def process_dataset(dataset="meld", max_k=12, top_n=1, vectordb_path=None, split=None, use_detailed_example=False, save_as="no", example_refinement_level=0):
    demonstration_creator = DemonstrationCreatorViaCache(vectordb_path, top_n=top_n, use_detailed_example=use_detailed_example,
                                                         example_refinement_level=example_refinement_level,
                                                         valid_emotion_set=get_valid_emotion_set_for_examples(dataset, example_refinement_level))

    splits = process_split(split)
    df = utils.get_dataset_as_dataframe(dataset, splits=splits)

    targets = {s: [] for s in splits}
    identifiers = {s: [] for s in splits}
    inputs = {s: [] for s in splits}

    df.fillna({"intensity_level":"medium"}, inplace=True)
    df.fillna({"pitch_level":"medium"}, inplace=True)
    df.fillna({"rate_level":"medium"}, inplace=True)

    # df["mapped_emotion"] = df["emotion"].map(emotion_mapper)
    df = df[["utterance", "emotion", "mapped_emotion", "idx", "split", "dialog_idx", "turn_idx", "speaker", "rate_level", "pitch_level", "intensity_level", "erc_target"]]
    df["abstracted_audio"] = df.apply(utils.abstacted_audio_text, axis=1)



    for split, df_split in df.groupby(by="split"):
        for conv_idx, df_conv in tqdm(df_split.groupby(by="dialog_idx"), desc=f"Processing conversations in split {split} for dataset {dataset}"):
            df_conv["speaker_idx"] = df_conv.groupby("speaker").ngroup()
            for unit in df_conv.to_dict(orient="records"):
                if dataset.lower() == "iemocap" and not unit["erc_target"]:
                    continue
                idx = unit["idx"]
                history_context = create_history_context(df_conv, unit["turn_idx"], max_k)
                speaker_id = unit["speaker"]
                utterance = unit["utterance"]
                audio_features = unit["abstracted_audio"]
                demonstrations = demonstration_creator.get_demonstration_text_via_idx(idx)
                target = unit["mapped_emotion"]

                inputs[split].append({"history":history_context, "utterance":utterance,
                                      "audio_features":audio_features, "speaker_id":speaker_id, "demonstrations":demonstrations})
                targets[split].append(target)
                identifiers[split].append(idx)


    dataset = {
        split: [{"input": x1, "target": x2, "idx": x3}
                for x1, x2, x3 in zip(inputs[split], targets[split], identifiers[split])]
        for split in splits
    }
    if save_as in ['json', 'jsonl']:
        data_path = paths.PROCESSED_DATA_DIR / dataset.upper() / f'k{max_k}_{demonstration_creator.get_id()}'
        data_path.mkdir(parents=True, exist_ok=True)
        if save_as == 'json':
            utils.save_dataset_as_json(dataset, data_path)
        else:
            utils.save_dataset_as_jsonl(dataset, data_path)
    return dataset


def main(config_dict=None):
    if config_dict is None:
        parser = argparse.ArgumentParser(description="Data processing script for LLM input")
        parser.add_argument("--dataset", type=str, default="iemocap", help="Dataset name to process, either meld or iemocap")
        parser.add_argument("--max_k", type=int, default=12, help="Window size for the history")
        parser.add_argument("--top_n", type=int, default=1, help="Number of utterance-emotion samples to retrieve for each llm input")
        parser.add_argument("--vectordb_path", type=str, default=DEFAULT_VECTORSTORE_PATH, help="Path to the vector database")
        parser.add_argument("--split", type=str, default=None, help="Choose the split to process of the dataset between train, test, and dev. Default is None which processes all splits")
        parser.add_argument("--use_detailed_example", type=lambda x: (str(x).lower() in ['true', '1', 't']), default=False, help="Map each utterance in example to an emotion if true for the examples from flow or hybrid db")
        parser.add_argument("--example_refinement_level", type=int, default=0, choices=[0,1,2], help="if 0, no refinement. If 1, retrieve examples from the same dataset. If 2, retrieve examples from the same dataset and all mapped emotions in the demonstration is in the valid emotions for the given dataset")
        parser.add_argument("--save_as", type=str, default="no", choices=["json", "jsonl", "no"], help="Save the processed data as json or jsonl. If no, do not save the data and return it.")
        args = parser.parse_args()
    else:
        defaults = {
            "split": "dev",
            "use_detailed_example": False,
            "example_refinement_level": 0,
            "save_as": "no"
        }
        defaults.update(config_dict)
        args = argparse.Namespace(**defaults)

    return process_dataset(dataset=args.dataset, max_k=args.max_k, top_n=args.top_n, vectordb_path=args.vectordb_path,
                    split=args.split, use_detailed_example=args.use_detailed_example, save_as=args.save_as,
                           example_refinement_level=args.example_refinement_level)

if __name__ == '__main__':
    main()