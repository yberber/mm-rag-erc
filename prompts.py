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
Please output only the selected emotion label and make no explanation:
"""

EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=EMOTION_RECOGNITION_TEMPLATE
)


EMOTION_RECOGNITION_CHAT_PROMPT = ChatPromptTemplate.from_template(EMOTION_RECOGNITION_TEMPLATE)


GEMINI_EMOTION_RECOGNITION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled AI expert in multimodal emotion recognition. Your task is to analyze the provided conversation, a specific target utterance, and its audio features to determine the speaker's emotion from a given list of options.

<|eot_id|><|start_header_id|>user<|end_header_id|>
<task_definition>
Analyze the data below to identify the emotion of the speaker in the target utterance.
</task_definition>

<examples>
{top_n_rag_examples}
</examples>

<conversation_context>
<history>
{history}
</history>
<target_utterance>
  <speaker>{speaker_id}</speaker>
  <text>{utterance}</text>
  <audio_features>{audio_features}</audio_features>
</target_utterance>
</conversation_context>

<emotion_options>
{emotion_set}
</emotion_options>

Based on all the provided information, what is the emotion of the target utterance? Provide your answer inside <emotion> XML tags. Do not include any other text or explanation.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<emotion>
"""


GEMINI_EMOTION_RECOGNITION_PROMPT= PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=GEMINI_EMOTION_RECOGNITION_TEMPLATE
)



CLAUDE_EMOTION_RECOGNITION_TEMPLATE = """You are an expert in sentiment and emotion analysis. Your task is to identify the emotion expressed in a specific utterance based on conversation context and audio features.

## Examples:
{top_n_rag_examples}

## Conversation History:
{history}

## Task:
Analyze the following utterance and identify its emotion:
- Speaker: {speaker_id}
- Utterance: "{utterance}"
- Audio Features: {audio_features}

## Available Emotion Labels:
{emotion_set}

## Instructions:
1. Consider the conversation context and how it relates to this utterance
2. Take into account the audio features provided
3. Select the most appropriate emotion label from the available options
4. Output ONLY the emotion label with no additional text or explanation

Emotion Label:"""

CLAUDE_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"],
    template=CLAUDE_EMOTION_RECOGNITION_TEMPLATE
)



GPT5_EMOTION_RECOGNITION_TEMPLATE = """
You are a careful emotion classifier.

TASK
- Pick exactly ONE label from the allowed set.
- Do NOT invent new labels.
- Answer with ONLY the label text, nothing else.

ALLOWED_LABELS
<{emotion_set}>

DECISION POLICY
- Prioritize the current utterance first, then conversation history, then audio cues.
- If cues conflict, prefer the utterance's literal meaning.
- If uncertainty remains between two or more labels, choose the most non-committal one present in ALLOWED_LABELS
  (e.g., "neutral" or "other"/"unknown" if provided).

CONTEXT
Few-shot examples:
{top_n_rag_examples}

Conversation (delimited by ###):
###
{history}
###

TARGET
Speaker: "{speaker_id}"
Utterance: "{utterance}"
Audio descriptors: {audio_features}

OUTPUT
Return ONLY one label from ALLOWED_LABELS.
"""

GPT5_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"],
    template=GPT5_EMOTION_RECOGNITION_TEMPLATE
)


GPT5_JSON_EMOTION_RECOGNITION_PROMPT = """
You are a careful emotion classifier.

INSTRUCTIONS
- Output valid minified JSON: {{"label":"<one_of_allowed_labels>"}} and nothing else.
- Do NOT add explanations or extra fields.
- Do NOT invent new labels.

ALLOWED_LABELS
<{emotion_set}>

DECISION POLICY
- Rank evidence: utterance > conversation history > audio descriptors.
- If mixed signals: prefer the utterance.
- If still uncertain: choose the least specific valid label available (e.g., "neutral", "other", or "unknown", if present).

FEW-SHOT
{top_n_rag_examples}

CONVERSATION (### delimited)
###
{history}
###

TARGET
Speaker: "{speaker_id}"
Utterance: "{utterance}"
Audio: {audio_features}

OUTPUT
{{"label":"<one_of_allowed_labels>"}}
"""

GPT5_JSON_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"],
    template=GPT5_JSON_EMOTION_RECOGNITION_PROMPT
)




GEMINI_2_EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert of sentiment and emotional analysis.

## Example(s)
{top_n_rag_examples}

## Conversation History
The following conversation noted between ‘######’ involves several speakers.
{history}

## Target Utterance
Speaker: {speaker_id}
Utterance: {{utterance}}
Acoustic Features: {{audio_features}}

## Task
Based on the conversation history and the utterance's acoustic features, please select the emotional label from <{emotion_set}>:

Target:
"""

GEMINI_2_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=GEMINI_2_EMOTION_RECOGNITION_TEMPLATE
)


CLAUDE_2_EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert of sentiment and emotional analysis.

Example(s):
{top_n_rag_examples}

The following conversation noted between '### ###' involves several speakers.
###
{history}
###

Based on the above historical utterances and acoustic information, please select the emotional label of the target utterance from <{emotion_set}>.

Target utterance: < "{speaker_id}" : "{utterance}" >
Acoustic features: {audio_features}

Please output only the selected emotion label and make no explanation:
"""

CLAUDE_2_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=CLAUDE_2_EMOTION_RECOGNITION_TEMPLATE
)


GEMINI_3_EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert of sentiment and emotional analysis.

## Example(s)
{top_n_rag_examples}

## Conversation History
{history}

## Target for Analysis
- Speaker: "{speaker_id}"
- Utterance: "{utterance}"
- Acoustic Features: "{audio_features}"

## Task
Based on the conversation history and the target's acoustic features, select the emotional label from <{emotion_set}>.
Please output only the selected emotion label and make no explanation:
"""

GEMINI_3_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=GEMINI_3_EMOTION_RECOGNITION_TEMPLATE
)


GPT_5_2_EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert of sentiment and emotional analysis.
<EXAMPLES>
{top_n_rag_examples}
</EXAMPLES>

The following conversation noted between '### ###' involves several speakers.
###
{history}
###
Based on the above historical utterances, select the emotion of the current utterance.

<UTTERANCE speaker="{speaker_id}">{utterance}</UTTERANCE>

<AUDIO "{audio_features}" />

Choose ONE label from <{emotion_set}>.
Output ONLY the label, nothing else.
"""

GPT_5_2_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["top_n_rag_examples", "history", "speaker_id", "utterance", "audio_features", "emotion_set"], template=GPT_5_2_EMOTION_RECOGNITION_TEMPLATE
)