from langchain_core.prompts.prompt import PromptTemplate


EMOTION_RECOGNITION_TEMPLATE = """Now you are an expert in dialogue emotion recognition.

Example(s):
{demonstrations}

The following conversation noted between '### ###' involves several speakers.
###
{history}
###

Based on the above conversation context, please select the emotional label of < "{speaker_id}" : "{utterance}" >, said with < {audio_features} >, from < {candidate_emotions} >.
Please output only the selected emotion label and make no explanation:
"""

# The corresponding PromptTemplate object
EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=EMOTION_RECOGNITION_TEMPLATE
)
EMOTION_RECOGNITION_PROMPT.name = "EMOTION_RECOGNITION_PROMPT"

GEMINI_EMOTION_RECOGNITION_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled AI expert in multimodal emotion recognition. Your task is to analyze the provided conversation, a specific target utterance, and its audio features to determine the speaker's emotion from a given list of options.

<|eot_id|><|start_header_id|>user<|end_header_id|>
<task_definition>
Analyze the data below to identify the emotion of the speaker in the target utterance.
</task_definition>

<examples>
{demonstrations}
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
{candidate_emotions}
</emotion_options>

Based on all the provided information, what is the emotion of the target utterance? Provide your answer inside <emotion> XML tags. Do not include any other text or explanation.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<emotion>
"""


GEMINI_EMOTION_RECOGNITION_PROMPT= PromptTemplate(
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"], template=GEMINI_EMOTION_RECOGNITION_TEMPLATE
)
GEMINI_EMOTION_RECOGNITION_PROMPT.name = "GEMINI_EMOTION_RECOGNITION_PROMPT"



CLAUDE_EMOTION_RECOGNITION_TEMPLATE = """You are an expert in sentiment and emotion analysis. Your task is to identify the emotion expressed in a specific utterance based on conversation context and audio features.

## Examples:
{demonstrations}

## Conversation History:
{history}

## Task:
Analyze the following utterance and identify its emotion:
- Speaker: {speaker_id}
- Utterance: "{utterance}"
- Audio Features: {audio_features}

## Available Emotion Labels:
{candidate_emotions}

## Instructions:
1. Consider the conversation context and how it relates to this utterance
2. Take into account the audio features provided
3. Select the most appropriate emotion label from the available options
4. Output ONLY the emotion label with no additional text or explanation

Emotion Label:"""

CLAUDE_EMOTION_RECOGNITION_PROMPT = PromptTemplate(
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=CLAUDE_EMOTION_RECOGNITION_TEMPLATE
)
CLAUDE_EMOTION_RECOGNITION_PROMPT.name = "CLAUDE_EMOTION_RECOGNITION_PROMPT"



GPT5_EMOTION_RECOGNITION_TEMPLATE = """
You are a careful emotion classifier.

TASK
- Pick exactly ONE label from the allowed set.
- Do NOT invent new labels.
- Answer with ONLY the label text, nothing else.

ALLOWED_LABELS
<{candidate_emotions}>

DECISION POLICY
- Prioritize the current utterance first, then conversation history, then audio cues.
- If cues conflict, prefer the utterance's literal meaning.
- If uncertainty remains between two or more labels, choose the most non-committal one present in ALLOWED_LABELS
  (e.g., "neutral" or "other"/"unknown" if provided).

CONTEXT
Few-shot examples:
{demonstrations}

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
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=GPT5_EMOTION_RECOGNITION_TEMPLATE
)

GPT5_EMOTION_RECOGNITION_PROMPT.name = "GPT5_EMOTION_RECOGNITION_PROMPT"








EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE = """Now you are an expert in dialogue emotion recognition.

The following conversation noted between '### ###' involves several speakers.
###
{history}
###

Based on the above conversation context, please select the emotional label of < "{speaker_id}" : "{utterance}" >, said with < {audio_features} >, from < {candidate_emotions} >.
Please output only the selected emotion label and make no explanation:
"""

# The corresponding PromptTemplate object
EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE = PromptTemplate(
    input_variables=["history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE
)
EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE.name = "EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE"

GEMINI_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a highly skilled AI expert in multimodal emotion recognition. Your task is to analyze the provided conversation, a specific target utterance, and its audio features to determine the speaker's emotion from a given list of options.

<|eot_id|><|start_header_id|>user<|end_header_id|>
<task_definition>
Analyze the data below to identify the emotion of the speaker in the target utterance.
</task_definition>

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
{candidate_emotions}
</emotion_options>

Based on all the provided information, what is the emotion of the target utterance? Provide your answer inside <emotion> XML tags. Do not include any other text or explanation.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<emotion>
"""


GEMINI_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE = PromptTemplate(
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=GEMINI_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE
)
GEMINI_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE.name = "GEMINI_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE"



CLAUDE_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE = """You are an expert in sentiment and emotion analysis. Your task is to identify the emotion expressed in a specific utterance based on conversation context and audio features.

## Conversation History:
{history}

## Task:
Analyze the following utterance and identify its emotion:
- Speaker: {speaker_id}
- Utterance: "{utterance}"
- Audio Features: {audio_features}

## Available Emotion Labels:
{candidate_emotions}

## Instructions:
1. Consider the conversation context and how it relates to this utterance
2. Take into account the audio features provided
3. Select the most appropriate emotion label from the available options
4. Output ONLY the emotion label with no additional text or explanation

Emotion Label:"""

CLAUDE_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE = PromptTemplate(
    input_variables=["history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=CLAUDE_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE
)
CLAUDE_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE.name = "CLAUDE_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE"



GPT5_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE = """
You are a careful emotion classifier.

TASK
- Pick exactly ONE label from the allowed set.
- Do NOT invent new labels.
- Answer with ONLY the label text, nothing else.

ALLOWED_LABELS
<{candidate_emotions}>

DECISION POLICY
- Prioritize the current utterance first, then conversation history, then audio cues.
- If cues conflict, prefer the utterance's literal meaning.
- If uncertainty remains between two or more labels, choose the most non-committal one present in ALLOWED_LABELS
  (e.g., "neutral" or "other"/"unknown" if provided).

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

GPT5_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE = PromptTemplate(
    input_variables=["demonstrations", "history", "speaker_id", "utterance", "audio_features", "candidate_emotions"],
    template=GPT5_EMOTION_RECOGNITION_TEMPLATE_WITHOUT_EXAMPLE
)

GPT5_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE.name = "GPT5_EMOTION_RECOGNITION_PROMPT_WITHOUT_EXAMPLE"