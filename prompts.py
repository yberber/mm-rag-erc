from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

prompt = PromptTemplate.from_template("Say {foo}")
x = prompt.format(foo="bar")
type(x)

EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert of sentiment and emotional analysis.
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

EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=EMOTION_RECOGNITION_TEMPLATE
)


EMOTION_RECOGNITION_CHAT_PROMPT = ChatPromptTemplate.from_template(EMOTION_RECOGNITION_TEMPLATE)


