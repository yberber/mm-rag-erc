
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm
import time

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.1:8b")
chain = prompt | model
chain.invoke({"question": "What is LangChain?"})


import json
iemocap_data = []
with open("dat/d.json") as file:
    for row in file:
        iemocap_data.append(json.loads(row))
iemocap_data[0]

emotion_set = ["joyful", "sad", "neutral", "angry", "excited", "frustrated"]
emotion_set = ", ".join(emotion_set)


from langchain_core.prompts.prompt import PromptTemplate
EMOTION_RECOGNITION_TEMPLATE1 = """Now you are an expert of sentiment and emotional analysis.
Example(s):
{top_n_rag_examples}
The following conversation noted between '### ###' involves several speakers.
###
{history}
###
Based on the above historical utterances, please select the emotional label of < "{speaker_id}" : "{utterance}" >, said
with {audio_features}, from <{emotion_set}>.
Please output only the selected emotion label:
"""
EMOTION_RECOGNITION_PROMPT1 = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=EMOTION_RECOGNITION_TEMPLATE1
)


EMOTION_RECOGNITION_TEMPLATE2 = """Now you are an expert of sentiment and emotional analysis.
Example(s):
{top_n_rag_examples}
The following conversation noted between '### ###' involves several speakers.
###
{history}
###
Based on the above historical utterances, please select the emotional label of < "{speaker_id}" : "{utterance}" >, said
with {audio_features}, from <{emotion_set}>.
Respond with just one label:
"""


EMOTION_RECOGNITION_PROMPT2 = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=EMOTION_RECOGNITION_TEMPLATE2
)


chain1 = EMOTION_RECOGNITION_PROMPT1 | model
chain2 = EMOTION_RECOGNITION_PROMPT2 | model



start_time = time.time()


wrong_output_cnt = 0
for unit in tqdm(iemocap_data[:10]):
    model_input = unit["input"]
    target = unit["target"]

    # print(EMOTION_RECOGNITION_PROMPT.invoke(
    #     {"top_n_rag_examples": model_input["example"],
    #      "history": model_input["history_context"],
    #      "speaker_id": model_input["speaker_id"],
    #      "utterance": model_input["utterance"],
    #      "audio_features": model_input["audio_features"],
    #      "emotion_set": emotion_set}
    # ))
    predicted = chain2.invoke({"top_n_rag_examples": model_input["example"],
                  "history": model_input["history_context"],
                  "speaker_id": model_input["speaker_id"],
                  "utterance": model_input["utterance"],
                  "audio_features": model_input["audio_features"],
                  "emotion_set": emotion_set})
    # print(f"Utterance: {model_input['utterance']}")
    print(f"Predicted: {predicted}, Target: {target}")
    if " " in predicted.strip():
        wrong_output_cnt+=1
        print("Wrong output count:", wrong_output_cnt)
    print("\n")

end_time = time.time()
print(f"Time taken {(end_time - start_time)}")


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(EMOTION_RECOGNITION_TEMPLATE2)
chain12 = prompt | model



start_time = time.time()
wrong_output_cnt = 0
for unit in tqdm(iemocap_data[:10]):
    model_input = unit["input"]
    target = unit["target"]

    predicted = chain12.invoke({"top_n_rag_examples": model_input["example"],
                  "history": model_input["history_context"],
                  "speaker_id": model_input["speaker_id"],
                  "utterance": model_input["utterance"],
                  "audio_features": model_input["audio_features"],
                  "emotion_set": emotion_set})
    print(f"Predicted: {predicted}, Target: {target}")
    if " " in predicted.strip():
        wrong_output_cnt+=1
        print("Wrong output count:", wrong_output_cnt)
    print("\n")

end_time = time.time()
print(f"Time taken {(end_time - start_time)}")
