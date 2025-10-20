import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from utils import (set_pandas_display_options, load_json, load_dataframe_from_json,
                   get_dataset_as_dataframe, str2bool)
import utils
set_pandas_display_options()

class DemonstrationCreatorViaCache:
    def __init__(self, vectordb_path, top_n=1, use_detailed_example=False):
        path_splitted = str(vectordb_path).split("/")
        path_splitted[-2] = "caches"

        possible_cache_path = "/".join(path_splitted)
        self.cache_data = load_json(path=str(possible_cache_path)+".json")
        self.top_n = top_n

        vectordb_name = str(vectordb_path).split("/")[-1]
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
                                                                    use_detailed_example=self.use_detailed_example)
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

        if idx is not None:
            excluded_prefixes.add(idx[:idx.rfind("_")])

        for ex_idx in examples_idx:
            idx_prefix = ex_idx[:ex_idx.rfind("_")]
            if idx_prefix not in excluded_prefixes:
                top_n.append(ex_idx)
                if len(top_n) >= self.top_n:
                    break
                excluded_prefixes.add(idx_prefix)

        return top_n



    def get_idx_to_speaker_utterance_emotion_df(self):
        return load_dataframe_from_json("vectorstore/idx_to_speaker_utterance_emotion_df.json")

    @staticmethod
    def get_demonstration_text_for_n1(df, turn_idx_target, turn_idx_start=None, type="single", use_detailed_example=False):
        if type == "single":
            row = df.loc[turn_idx_target]
            return f'< {row["utterance"]} : {row["mapped_emotion"]} >'
        elif type in ["flow", "hybrid"]:
            df = df.loc[turn_idx_start:turn_idx_target]
            page_content_lines = []
            for _, row in df.iterrows():
                line = f"{row['speaker']} : {row['utterance']}"
                if use_detailed_example:
                    line += f" :-> {row['mapped_emotion']}"
                page_content_lines.append(line)
            demonstration = "\n".join(page_content_lines)
            if use_detailed_example:
                return f"<\n{demonstration}\n>"
            else:
                return f"< {demonstration}\nLast utterance label is: {df.iloc[-1]['mapped_emotion']} >"
        else:
            raise Exception(f"Unknown type: {type}")





    # @staticmethod
    # def get_examples_cached(cache_dict, top_n, idx, idx_to_utterance_emotion_dict, emotion_set):
    #     examples_idx = cache_dict[idx]
    #     example_text = ""
    #     emotion_count = 0
    #     for ex_idx in examples_idx:
    #         utterance, emotion = idx_to_utterance_emotion_dict[ex_idx]
    #         emotion = emotion_mapper_union.get(emotion, None)
    #         if emotion not in emotion_set:
    #             continue
    #         example_text += f"< {utterance} : {emotion} >"
    #         emotion_count += 1
    #         if emotion_count >= top_n:
    #             break
    #     return example_text.strip('\n')

    # @staticmethod
    # def get_examples_via_db(db, utterance, max_k, idx=None):
    #     out = db.similarity_search(utterance, k=max_k + 1)
    #     if idx:
    #         out = [o for o in out if o.metadata["idx"] != idx]
    #     out = out[:max_k]
    #     example_text = ""
    #     for sim_doc in out:
    #         example_text += f"< {sim_doc.page_content} : {sim_doc.metadata['emotion']} >\n"
    #     return example_text.strip('\n')





def abstacted_audio_text(row):
    return (f"{row['rate_level']} speech rate, {row['pitch_level']} "
            f"pitch, and {row['intensity_level']} intensity")

def create_history_context(conversation, current_utterance_idx, max_k):
    context = ""
    conversation_history = conversation[max(0, current_utterance_idx - max_k):current_utterance_idx+1]
    for unit in conversation_history.to_dict(orient="records"):
        context += f"{unit['speaker']}: {unit['utterance']}\n"
    return context.strip("\n")



def get_vectordb_instance(path_to_db):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        collection_name=str(path_to_db).split("/")[-1],
        embedding_function=embeddings,
        persist_directory=path_to_db,
    )
    return vector_db




def process_dataset(dataset="meld", max_k=12, top_n=1, vectordb_path="vectorstore/vectorstore_db/meld_iemocap_single", split=None, use_detailed_example=False):

    if isinstance(use_detailed_example, str):
        if use_detailed_example.lower() == "true":
            use_detailed_example = True
        elif use_detailed_example.lower() == "false":
            use_detailed_example = False
        else:
            raise ValueError(f"use_detailed_example must be either True or False, but got {use_detailed_example}")


    demonstration_creator = DemonstrationCreatorViaCache(vectordb_path, top_n=top_n, use_detailed_example=use_detailed_example)



    df = get_dataset_as_dataframe(dataset, splits=split)



    splits = [split] if split else df["split"].unique().tolist()

    targets = {s: [] for s in splits}
    identifiers = {s: [] for s in splits}
    inputs = {s: [] for s in splits}




    df.fillna({"intensity_level":"medium"}, inplace=True)
    df.fillna({"pitch_level":"medium"}, inplace=True)
    df.fillna({"rate_level":"medium"}, inplace=True)

    # df["mapped_emotion"] = df["emotion"].map(emotion_mapper)
    df = df[["utterance", "emotion", "mapped_emotion", "idx", "split", "dialog_idx", "turn_idx", "speaker", "rate_level", "pitch_level", "intensity_level", "erc_target"]]
    df["abstracted_audio"] = df.apply(abstacted_audio_text, axis=1)



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
                # input = EMOTION_RECOGNITION_PROMPT.format(top_n_rag_examples=example, history=history_context,
                #                                   speaker_id=speaker_id, utterance=utterance,
                #                                   audio_features=audio_features, emotion_set=emotion_set_text)
                target = unit["mapped_emotion"]

                inputs[split].append({"history":history_context, "utterance":utterance,
                                      "audio_features":audio_features, "speaker_id":speaker_id, "demonstrations":demonstrations})

                # input_prompts[split].append(input)
                targets[split].append(target)
                identifiers[split].append(idx)

    data_path = f'PROCESSED_DATASET/{dataset.upper()}/k{max_k}_{demonstration_creator.get_id()}'

    os.makedirs(data_path, exist_ok=True)
    for split in splits:
        data_to_save = list(zip(inputs[split], targets[split], identifiers[split]))
        data_to_save = [{"input": x1, "target": x2, "idx" : x3} for (x1, x2, x3) in data_to_save]
        utils.save_as_json(f"{data_path}/{split}.json", data_to_save)
        # with open(f'{data_path}/{split}.json', 'w') as file:
        #     for input, target, identifier in zip(inputs[split], targets[split], identifiers[split]):
        #         file.write(json.dumps({'input': input, 'target': target, 'identifier': identifier}) + '\n')
    print(f"Done.")

def main(config_dict=None):
    if config_dict is None:
        parser = argparse.ArgumentParser(description="Data processing script for LLM input")
        parser.add_argument("--dataset", type=str, default="iemocap", help="Dataset name to process, either meld or iemocap")
        parser.add_argument("--max_k", type=int, default=12, help="Window size for the history")
        parser.add_argument("--top_n", type=int, default=1, help="Number of utterance-emotion samples to retrieve for each llm input")
        parser.add_argument("--vectordb_path", type=str, default="vectorstore/vectorstore_db/meld_iemocap_single", help="Path to the vector database")
        parser.add_argument("--split", type=str, default=None, help="Choose the split to process of the dataset between train, test, and dev. Default is None which processes all splits")
        parser.add_argument("--use_detailed_example", type=str, default="False", help="Map each utterance in example to an emotion if true for the examples from flow or hybrid db")
        args = parser.parse_args()
    else:
        args = argparse.Namespace(**config_dict)

    process_dataset(dataset=args.dataset, max_k=args.max_k, top_n=args.top_n, vectordb_path=args.vectordb_path,
                    split=args.split, use_detailed_example=str2bool(args.use_detailed_example))

if __name__ == '__main__':
    main()